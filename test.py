from nemo.collections.nlp.data.dialogue_state_tracking.sgd.input_example import InputExample
from nemo.collections.nlp.models.dialogue_state_tracking.sgdqa_model import SGDQAModel
from nemo.collections.nlp.data.dialogue_state_tracking import Schema, SGDDataProcessor, SGDDataset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import os
import json
import torch
import copy
import re
import pickle
import argparse

NUM_TASKS = 6

# Adapted from 'eval_step_helper' in
# NeMo/nemo/collections/nlp/models/dialogue_state_tracking/sgdqa_model.py
def predict(model, device, batch):
    (
        example_id_num,
        service_id,
        utterance_ids,
        token_type_ids,
        attention_mask,
        start_char_idx,
        end_char_idx
    ) = batch

    utterance_ids = utterance_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_spans,
        ) = model(input_ids=utterance_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    all_example_id_num = []
    all_service_id = []
    all_logit_intent_status = []
    all_logit_req_slot_status = []
    all_logit_cat_slot_status = []
    all_logit_cat_slot_value_status = []
    all_logit_noncat_slot_status = []
    all_logit_spans = []
    all_start_char_idx = []
    all_end_char_idx = []

    all_example_id_num.append(example_id_num)
    all_service_id.append(service_id)
    all_logit_intent_status.append(logit_intent_status)
    all_logit_req_slot_status.append(logit_req_slot_status)
    all_logit_cat_slot_status.append(logit_cat_slot_status)
    all_logit_cat_slot_value_status.append(logit_cat_slot_value_status)
    all_logit_noncat_slot_status.append(logit_noncat_slot_status)
    all_logit_spans.append(logit_spans)
    all_start_char_idx.append(start_char_idx)
    all_end_char_idx.append(end_char_idx)

    # after this: all_x is list of tensors, of length world_size
    example_id_num = torch.cat(all_example_id_num)
    service_id = torch.cat(all_service_id)
    logit_intent_status = torch.cat(all_logit_intent_status)
    logit_req_slot_status = torch.cat(all_logit_req_slot_status)
    logit_cat_slot_status = torch.cat(all_logit_cat_slot_status)
    logit_cat_slot_value_status = torch.cat(all_logit_cat_slot_value_status)
    logit_noncat_slot_status = torch.cat(all_logit_noncat_slot_status)
    logit_spans = torch.cat(all_logit_spans)
    start_char_idx = torch.cat(all_start_char_idx)
    end_char_idx = torch.cat(all_end_char_idx)

    intent_status = torch.nn.Sigmoid()(logit_intent_status)

    # Scores are output for each requested slot.
    req_slot_status = torch.nn.Sigmoid()(logit_req_slot_status)

    # For categorical slots, the status of each slot and the predicted value are output.
    cat_slot_status_dist = torch.nn.Softmax(dim=-1)(logit_cat_slot_status)

    cat_slot_status = torch.argmax(logit_cat_slot_status, axis=-1)
    cat_slot_status_p = torch.max(cat_slot_status_dist, axis=-1)[0]
    cat_slot_value_status = torch.nn.Sigmoid()(logit_cat_slot_value_status)

    # For non-categorical slots, the status of each slot and the indices for spans are output.
    noncat_slot_status_dist = torch.nn.Softmax(dim=-1)(logit_noncat_slot_status)

    noncat_slot_status = torch.argmax(logit_noncat_slot_status, axis=-1)
    noncat_slot_status_p = torch.max(noncat_slot_status_dist, axis=-1)[0]

    softmax = torch.nn.Softmax(dim=1)

    scores = softmax(logit_spans)
    start_scores, end_scores = torch.unbind(scores, dim=-1)

    batch_size, max_num_tokens = end_scores.size()
    # Find the span with the maximum sum of scores for start and end indices.
    total_scores = torch.unsqueeze(start_scores, axis=2) + torch.unsqueeze(end_scores, axis=1)
    start_idx = torch.arange(max_num_tokens, device=total_scores.get_device()).view(1, -1, 1)
    end_idx = torch.arange(max_num_tokens, device=total_scores.get_device()).view(1, 1, -1)
    invalid_index_mask = (start_idx > end_idx).repeat(batch_size, 1, 1)
    total_scores = torch.where(
        invalid_index_mask,
        torch.zeros(total_scores.size(), device=total_scores.get_device(), dtype=total_scores.dtype),
        total_scores,
    )
    max_span_index = torch.argmax(total_scores.view(-1, max_num_tokens ** 2), axis=-1)
    max_span_p = torch.max(total_scores.view(-1, max_num_tokens ** 2), axis=-1)[0]

    span_start_index = torch.floor_divide(max_span_index, max_num_tokens)
    span_end_index = torch.fmod(max_span_index, max_num_tokens)

    tensors = {
        'example_id_num': example_id_num,
        'service_id': service_id,
        'intent_status': intent_status,
        'req_slot_status': req_slot_status,
        'cat_slot_status': cat_slot_status,
        'cat_slot_status_p': cat_slot_status_p,
        'cat_slot_value_status': cat_slot_value_status,
        'noncat_slot_status': noncat_slot_status,
        'noncat_slot_status_p': noncat_slot_status_p,
        'noncat_slot_p': max_span_p,
        'noncat_slot_start': span_start_index,
        'noncat_slot_end': span_end_index,
        'noncat_alignment_start': start_char_idx,
        'noncat_alignment_end': end_char_idx,
    }
    return tensors

# Changle Multiwoz dialogue id format to SGD-like
def format_dialog_id(dialog_id):
    dialog_id = dialog_id.replace('.json', '')
    id_mapping = {'MUL': 100, 'SNG': 101, 'PMUL': 102}
    alpha = re.findall('[A-Z]+', dialog_id)
    if len(alpha) != 0 and alpha[0] in alpha:
        dialog_id = dialog_id.replace(alpha[0], f'{id_mapping[alpha[0]]}_')
    
    dialog_id_1 = dialog_id.split('_')[1]
    dialog_id = dialog_id.replace(dialog_id_1, dialog_id_1.zfill(5))
    return dialog_id

# Adapted from '_create_examples_from_dialogue'
# NeMo/nemo/collections/nlp/models/dialogue_state_tracking/sgdqa_model.py 
def create_examples_from_dialogue(dialogue, dialogues_processor, dataset_split, schemas):
    dial_turn_examples = []
    dialog_id = format_dialog_id(dialogue['dialogue_id'])
    for turn_idx, turn in enumerate(dialogue['turns']):
        if turn['speaker'] == 'SYSTEM':
            continue

        turn_id = "{}-{}-{:02d}".format(dataset_split, dialog_id, turn_idx)
        if turn_idx == 0:
            system_utterance = ''
        else:
            system_utterance = dialogue['turns'][turn_idx-1]['utterance']
        user_utterance = turn['utterance']

        basic_frame = {
            'slots': [],
            'state': {
                'active_intent': '',
                'slot_values': {},
                'requested_slots': [],
            },
        }
        system_frames = {}
        user_frames = {service: basic_frame for service in dialogue['services']}
        prev_states = {}
        turn_examples, prev_states, slot_carryover_values = dialogues_processor._create_examples_from_turn(
            turn_id,
            system_utterance,
            user_utterance,
            system_frames,
            user_frames,
            prev_states,
            schemas,
            True,
        )
        dial_turn_examples.extend(turn_examples)
    return dial_turn_examples

@hydra_runner(config_path="conf", config_name="sgdqa_config")
def main(cfg: DictConfig) -> None:
    model = SGDQAModel.from_pretrained(cfg.pretrained_model)
    all_schema_json_paths = [os.path.join(cfg.model.dataset.data_dir, 'schema.json')]
    schemas = Schema(all_schema_json_paths)
    schema_config = {
        "MAX_NUM_CAT_SLOT": cfg.model.dataset.max_num_cat_slot,
        "MAX_NUM_NONCAT_SLOT": cfg.model.dataset.max_num_noncat_slot,
        "MAX_NUM_VALUE_PER_CAT_SLOT": cfg.model.dataset.max_value_per_cat_slot,
        "MAX_NUM_INTENT": cfg.model.dataset.max_num_intent,
        "NUM_TASKS": NUM_TASKS,
        "MAX_SEQ_LENGTH": cfg.model.dataset.max_seq_length,
    }
    dialogues_processor = SGDDataProcessor(
        task_name=cfg.model.dataset.task_name,
        data_dir=cfg.model.dataset.data_dir,
        dialogues_example_dir=cfg.model.dataset.dialogues_example_dir,
        tokenizer=model.tokenizer,
        schemas=schemas,
        schema_config=schema_config,
        subsample=True,
    )

    data_dir = cfg.model.dataset.data_dir
    split = cfg.split
    task_name = 'sgd_all'
    input_json_files = SGDDataProcessor.get_dialogue_files(
        data_dir, split, task_name
    )

    debug = 0
    for filename in input_json_files:            
        with open(filename, 'r') as f:
            dialogues = json.load(f)
        turn_examples = []
        for dialogue in dialogues:
            examples = create_examples_from_dialogue(dialogue, dialogues_processor, split, schemas)
            turn_examples.extend(examples)
            if debug:
                break

        dataset = SGDDataset(turn_examples)
        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.model.test_ds.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.model.test_ds.drop_last,
            shuffle=cfg.model.test_ds.shuffle,
            num_workers=cfg.model.test_ds.num_workers,
            pin_memory=cfg.model.test_ds.pin_memory,
        )
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        preds = []
        for batch in tqdm(dl):
            pred = predict(model, device, batch)
            preds.append(pred)

        ids_to_service_names_dict = dialogues_processor.schemas._services_id_to_vocab
        model.multi_eval_epoch_end_helper(preds, split, dl, dialogues_processor, filename)
        if debug:
            break

if __name__ == '__main__':
    main()