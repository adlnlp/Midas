import transformers
from transformers import AutoTokenizer,AutoModel
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.llama.modeling_llama import add_start_docstrings_to_model_forward, LLAMA_INPUTS_DOCSTRING, Optional, List, BaseModelOutputWithPast, Union, Tuple, LlamaPreTrainedModel, SequenceClassifierOutputWithPast, LlamaModel
from train_dcidsfpos import logger
import types

'''
Implementation for unmask-Llama, doesn't support flashattention. transformers version is 4.35.0
Copied from transformers.mmodels.llama.modeling_llama.py
Used for unmask llama by replacing _prepare_4d_causal_attention_mask with transformers.modeling_attn_mask_utils._prepare_4d_attention_mask
'''
# if is_torch_fx_available():
#     # mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
_prepare_4d_attention_mask = transformers.modeling_attn_mask_utils._prepare_4d_attention_mask

@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        
        attention_mask = _prepare_4d_attention_mask(
            attention_mask, inputs_embeds.dtype
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

class LlamaForDCIDSFPOS(LlamaPreTrainedModel):
    def __init__(self, config, **mykwargs):
        super().__init__(config)
        self.model_name = mykwargs['name']
        self.freeze = mykwargs['freeze']
        self.head_cfg = mykwargs['head']
        self.mymask = mykwargs['mymask']
        
        self.model = LlamaModel(config)

        
        if self.mymask == False:
            self.model.forward = types.MethodType(forward, self.model)
                
        
        for param in self.model.parameters():
            param.requires_grad = not self.freeze

        # build head:
        assert self.head_cfg['type'] in ('dc','id','sf', 'pos','cdc'), "ERROR HEAD CFG!"
        self.classifier = nn.Linear(self.model.config.hidden_size,self.head_cfg['nclasses'])

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print("self.config.pad_token_id ",self.config.pad_token_id,self.config.eos_token_id)
        # print("output_hidden_states ",output_hidden_states,return_dict)
        # print(len(outputs),outputs.keys())
        # print(outputs[0].shape)
        sequence_output = outputs[0] # outputs[0] is the last hidden state
        logits = self.classifier(sequence_output)

        # sequence_output = self.dropout(sequence_output)
        if self.head_cfg['type'] in ('dc','id','cdc'):
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                        logits.device
                    )
                else:
                    sequence_lengths = -1

            if self.head_cfg['pooling'] == 'last':
                logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
                outputs.hidden_states += (outputs.hidden_states[-1][torch.arange(batch_size, device=logits.device), sequence_lengths],)
            elif self.head_cfg['pooling'] == 'mean':
                logits = torch.mean(logits, dim=1)
                outputs.hidden_states += (torch.mean(outputs.hidden_states[-1], dim=1),)
            else:
                raise NotImplementedError
        elif self.head_cfg['type'] in ('sf','pos'):
            pass

        loss = None
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if self.head_cfg['type'] in ('sf','pos'):
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )


class Llama(nn.Module):
    save_path = 'models/teacher'
    # for sequence classification using Llama, we havc 2 pooling method, last and mean
    # the official version in the huggingface uses last
    def __init__(self, name='Llama-2-7b-hf',head={'type':'sf','nclasses':2,'pooling':'last'},loracfg={'lora_r':64,'lora_alpha':16,'lora_dropout':0.1},freeze = True,load_ft=None,mymask=True) :
        super(Llama, self).__init__()
        self.model_name = name
        self.freeze = freeze
        self.head_cfg = head
        self.loracfg = loracfg
        self.load_ft = load_ft
        self.mymask = mymask
        
        if self.head_cfg['type'] in ('dc','id','cdc'):
            self.peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=self.loracfg['lora_r'], lora_alpha=self.loracfg['lora_alpha'], lora_dropout=self.loracfg['lora_dropout'])
        else:
            self.peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=self.loracfg['lora_r'], lora_alpha=self.loracfg['lora_alpha'], lora_dropout=self.loracfg['lora_dropout'])
        mykwargs = dict(name=self.model_name,head=self.head_cfg,freeze=self.freeze,mymask=mymask)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained('{}/{}'.format(self.save_path,self.model_name+"_tokenizer"))
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=' ')
            self.tokenizer.save_pretrained('{}/{}'.format(self.save_path,self.model_name+"_tokenizer"))

        self.tokenizer.add_special_tokens({'pad_token':self.tokenizer.eos_token})
        # need to specify the pad_id first. LLM need a specified pad token for sequence classification
        # A common way is using the eos token as the pad token
        try:
            self.model = LlamaForDCIDSFPOS.from_pretrained('{}/{}'.format(self.save_path,self.model_name),pad_token_id=self.tokenizer.pad_token_id,**mykwargs).bfloat16()
        except:
            self.model = LlamaForDCIDSFPOS.from_pretrained(self.model_name, token=' ',pad_token_id=self.tokenizer.pad_token_id,**mykwargs).bfloat16()
            self.model.save_pretrained('{}/{}'.format(self.save_path,self.model_name), from_pt=True)

        # self.model = LlamaForDCIDSFPOS(self.save_path, name=self.model_name,head=self.head_cfg,freeze=self.freeze)

        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

        if self.load_ft is not None:
            logger.info("Loading fine tuned state dict {}...".format(self.load_ft))
            sd = torch.load(self.load_ft)
            # neglect the pretrinaed head
            for name in list(sd.keys()):
                # print("saved sd ",name, name.startswith("model.base_model.model.classifier"))
                if name.startswith("model.base_model.model.classifier"):
                    # print("??????? Neglect")
                    logger.info("Neglect: {}".format(name))
                    sd.pop(name)
            self.load_state_dict(sd,strict=False)
        # print(self.state_dict().keys())
            # freeze again to freeze the lora parameters
            for name, param in self.named_parameters():
                if not name.startswith("model.base_model.model.classifier"):
                    # print("name ",name, param.requires_grad)
                    param.requires_grad = not self.freeze
                else:
                    logger.info("{}:{}".format(name, param.requires_grad))
        # print(self.model)

    def forward(self, inputs):
        out = self.model(**inputs,output_hidden_states=True,return_dict=True)
        # print(out)
        return out.hidden_states[-1], out.logits
    

# def __main__()