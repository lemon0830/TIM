# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss, MarginRankingLoss
import random
import copy

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, rl_alpha=0., Annealing_step=-1):
        super().__init__()

        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning

        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd

            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)

        self.case = -1

        if hasattr(base_model, "transformer"):
            self.case = 0
            self.rwmodel = base_model
        else:
            self.case = 1
            self.rwmodel = base_model

        self.PAD_ID = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.rl_alpha = rl_alpha

        self.Annealing_step = Annealing_step

        if self.Annealing_step > 0:
            self.step = 0.

    def gradient_checkpointing_enable(self):
        self.rwmodel.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwmodel.gradient_checkpointing_disable()

    def update_steps(self, step):
        self.step = step

    def rrhf_loss(self, scores, idxs, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1)  # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1)  # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        return -diff[aval].sum()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels = None,
                use_cache=False):

        seq_len = input_ids.shape[1] // 2

        input_ids = torch.cat([input_ids[:, :seq_len], input_ids[:, seq_len:]], dim=0)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask[:, :seq_len], attention_mask[:, seq_len:]], dim=0)

        labels = torch.cat([labels[:, :seq_len], labels[:, seq_len:]], dim=0)

        if self.case == 0:
            transformer_outputs = self.rwmodel.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache)
        elif self.case == 1:
            transformer_outputs = self.rwmodel.model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache)
        elif self.case == 2:
            transformer_outputs = self.rwmodel.cpmbee(
                input_ids)

        hidden_states = transformer_outputs[0]

        if hasattr(self.rwmodel, "lm_head"):
            lm_logits = self.rwmodel.lm_head(hidden_states)
        else:
            lm_logits = self.rwmodel.transformer.output_layer(hidden_states)
            lm_logits = lm_logits.transpose(0, 1).contiguous()
            hidden_states = hidden_states.transpose(0, 1)

        rewards = self.v_head(hidden_states).squeeze(-1)

        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        lm_logits = lm_logits[:bs]
        labels = labels[:bs]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        # Compute pairwise loss. Only backprop on the different tokens before padding
        r_loss = 0
        margin_rank_loss = MarginRankingLoss(margin=1.0)
        if self.rl_alpha > 0.:
            for i in range(bs):
                chosen_id = chosen_ids[i]
                rejected_id = rejected_ids[i]
                chosen_reward = chosen_rewards[i]
                rejected_reward = rejected_rewards[i]

                c_inds = (chosen_id == self.PAD_ID).nonzero()
                c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                    c_inds
                ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
                check_divergence = (chosen_id != rejected_id).nonzero()

                if len(check_divergence) == 0:
                    end_ind = rejected_reward.size(-1)
                    divergence_ind = end_ind - 1
                    r_ind = c_ind
                else:
                    # Check if there is any padding otherwise take length of sequence
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = r_inds[self.num_padding_at_beginning].item(
                    ) if len(r_inds) > self.num_padding_at_beginning else seq_len

                    end_ind = max(c_ind, r_ind)

                    divergence_ind = check_divergence[0]

                if divergence_ind <= 0:
                    continue
                assert divergence_ind > 0, f"duvergebce_ind {divergence_ind}. No match chosen_id {self.tokenizer.decode(chosen_id)} and rejected_id {self.tokenizer.decode(rejected_id)}"

                scores_A = chosen_reward[divergence_ind:end_ind]
                scores_B = rejected_reward[divergence_ind:end_ind]

                chosen_mean_scores.append(
                    chosen_reward[c_ind - 1])  #use the end score for reference
                rejected_mean_scores.append(rejected_reward[r_ind - 1])

                pos_AB = scores_A.repeat(len(scores_A), 1) > scores_B.repeat(len(scores_A), 1)

                loss_AB = margin_rank_loss(scores_A, scores_B, pos_AB.float())
                r_loss += loss_AB / len(scores_A)

            r_loss = r_loss / bs

            loss = loss + r_loss

        # print("good:", [i.item() for i in chosen_mean_scores])
        # print("bad :", [i.item() for i in rejected_mean_scores])

        if len(chosen_mean_scores) > 0:
            chosen_mean_scores = torch.stack(chosen_mean_scores)
            rejected_mean_scores = torch.stack(rejected_mean_scores)

        return {
            "logits": shift_logits,
            "labels": shift_labels,
            "loss": loss,
            "r_loss": r_loss.item() if self.rl_alpha > 0 else 0.,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        transformer_outputs = self.rwmodel(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }