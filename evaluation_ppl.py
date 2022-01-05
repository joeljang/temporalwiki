
from Datasets import CustomDataset
from torch.utils.data import DataLoader
import csv
import os
import torch
import nltk
from nltk.tokenize import word_tokenize

def evaluate_ppl(args, Model):
    model = Model(args)
    if args.checkpoint_path!="":
        model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)

    model.eval()
    model.to('cuda')
    loss_fct = torch.nn.CrossEntropyLoss()
    tokenizer = model.tokenizer
    #Get Validation Data
    if args.mode=='pretrain' or args.mode=='finetune' or args.mode=='evaluate_ppl' or args.mode== 'evaluate_ppl_corpus':
        dataset = CustomDataset(tokenizer, 'validation', input_length=args.max_input_length, 
                        output_length=args.max_output_length, args=args)
    else:
        raise Exception('Select the correct mode please.')
    print('Length of validation data: ',len(dataset))
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
    
    total_loss = 0
    batch_cnt = 0

    def clean_up(text):
        text =text.replace('<pad>', '')
        text = text.replace('</s>', '')
        text = text.replace(".", '')
        text = text.replace(',', '')
        text = text.replace("'", '')
        text = text.replace('"', '')
        return text   
    # If folder doesn't exist, then create it.
    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

    if args.mode == 'evaluate_ppl':
        with open(args.output_log, 'w', newline='') as writefile:  
            writer = csv.writer(writefile)
            for batch in iter(loader):
                with torch.no_grad():
                    lm_labels = batch['target_ids']
                    if 't5' in args.model_name_or_path:
                        print(model.ids_to_clean_text(batch['source_ids']))
                        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                        outputs = model.model(
                            input_ids=batch["source_ids"].cuda(),
                            attention_mask=batch["source_mask"].cuda(),
                            labels=lm_labels.cuda(),
                            decoder_attention_mask=batch['target_mask'].cuda(),
                        )
                    elif 'gpt2' in args.model_name_or_path:
                        # print(batch['source_nonprompt_mask'])
                        # print(batch["source_ids"])
                        # source_nonprompt_mask = batch['source_nonprompt_mask']
                        # lm_labels[source_nonprompt_mask == 0] = -100
                        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                        outputs = model.model(
                            input_ids=batch["source_ids"].cuda(),
                            attention_mask=batch["source_mask"].cuda(),
                            labels=lm_labels.cuda(),
                        )
                    loss = outputs[0]
                    total_loss += loss
                    batch_cnt += 1
                    # texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
                    # print(lm_labels)
                    # print(texts, targets)
                    print("perplexity", batch_cnt, loss.item(), torch.exp(total_loss/batch_cnt).item())
        with open(args.output_log, 'a', newline='') as writefile:  
            writer = csv.writer(writefile)
            writer.writerow([torch.exp(total_loss/batch_cnt).item()])
        print(f'Perplexity is: {torch.exp(total_loss/batch_cnt).item()}')
    
    elif args.mode == 'evaluate_ppl_corpus':
        tokenizer.padding_side = "right"
        with torch.no_grad():
            total_perplexity = 0
            total_loss = 0 
            total_ner_loss = 0
            total_noun_loss = 0
            total_verb_loss = 0
            total_num_loss = 0
            batch_idx = 0
            # total_new_loss = 0
            if 'gpt2' in args.model_name_or_path:
                for batch in iter(loader):
                    batch_idx+=1
                    lm_labels = batch["target_ids"].cuda()
                    ner_flags = torch.zeros_like(lm_labels)
                    noun_flags = torch.zeros_like(lm_labels)
                    verb_flags = torch.zeros_like(lm_labels)
                    num_flags = torch.zeros_like(lm_labels)
                    target = tokenizer.batch_decode(batch["target_ids"], clean_up_tokenization_spaces=False)
                    # print("Target list", target)
                    for index, target_seq in enumerate(target):
                        target_seq = target_seq.replace(' <pad>', '')
                        target_list = target_seq.split(' ')
                        mapping ={}
                        token_index = 0
                        word_index=0
                        for word in target_list:
                            if word == '':
                                if word_index!=0:
                                    token_index+=1
                                continue
                            
                            if word_index ==0:
                                word_token = tokenizer.batch_encode_plus([word])["input_ids"][0]
                            else: 
                                word_token = tokenizer.batch_encode_plus([word], add_prefix_space=True)["input_ids"][0]
                            mapping[word_index] = [token_index, token_index+ len(word_token)]
                            token_index+=len(word_token)
                            word_index+=1
                        # print(mapping)
                        target_list = list(filter(None, target_list))
                        pos_tag = nltk.pos_tag(target_list)
                        # print(pos_tag)
                        for pos_index, pos in enumerate(pos_tag):
                            # print(pos[0],tokenizer.batch_decode(lm_labels[index, mapping[pos_index][0]: mapping[pos_index][1]]))
                            if pos[1] == 'NNP' or pos[1] == 'NNPS':
                                ner_flags[index, mapping[pos_index][0]: mapping[pos_index][1]] = 1
                            elif pos[1] == 'NN' or pos[1] == 'NNS':
                                noun_flags[index, mapping[pos_index][0]: mapping[pos_index][1]] = 1
                            elif 'VB' in pos[1]:
                                verb_flags[index, mapping[pos_index][0]: mapping[pos_index][1]] = 1
                            elif 'CD' in pos[1]:
                                num_flags[index, mapping[pos_index][0]: mapping[pos_index][1]] = 1
                    
                    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

                    outputs = model.model(
                        input_ids=batch["source_ids"].cuda(),
                        attention_mask=batch["source_mask"].cuda(),
                        labels=lm_labels,
                    )
                    logits = outputs[1]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = lm_labels[..., 1:].contiguous()
                    ner_flags = ner_flags[..., 1:].contiguous()
                    noun_flags = noun_flags[..., 1:].contiguous()
                    verb_flags = verb_flags[..., 1:].contiguous()
                    num_flags = num_flags[..., 1:].contiguous()

                    ner_label = shift_labels.clone().detach()
                    noun_label = shift_labels.clone().detach()
                    verb_label = shift_labels.clone().detach()
                    num_label = shift_labels.clone().detach()
                    ner_label[ner_flags[:, :] == 0] = -100
                    noun_label[noun_flags[:,:] == 0] = -100
                    verb_label[verb_flags[:,:] == 0] = -100
                    num_label[num_flags[:,:] == 0] = -100

                    loss_ner = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), ner_label.view(-1)) 
                    loss_noun = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), noun_label.view(-1)) 
                    loss_verb = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), verb_label.view(-1)) 
                    loss_num = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), num_label.view(-1)) 
                    new_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) 

                
                    loss = outputs[0]
                    print("loss", loss, new_loss)

                    total_loss += loss
                    total_ner_loss += loss_ner
                    total_noun_loss += loss_noun
                    total_verb_loss += loss_verb
                    total_num_loss += loss_num
                    # total_new_loss += new_loss

                    # average_new_loss = total_new_loss / batch_idx
                    average_loss = total_loss / batch_idx
                    average_ner_loss = total_ner_loss / batch_idx
                    average_noun_loss = total_noun_loss / batch_idx
                    average_verb_loss = total_verb_loss / batch_idx
                    average_num_loss = total_num_loss / batch_idx

                    average_perplexity = torch.exp(average_loss)
                    average_ner_perplexity = torch.exp(average_ner_loss)
                    average_noun_perplexity = torch.exp(average_noun_loss)
                    average_verb_perplexity = torch.exp(average_verb_loss)
                    average_num_perplexity = torch.exp(average_num_loss)

                    print(batch_idx, " perplexity is ",average_perplexity.item(), average_ner_perplexity.item(), average_noun_perplexity.item(), average_verb_perplexity.item(), average_num_perplexity.item())
                    # print(batch_idx, " perplexity is ",average_perplexity)
                        
                with open(args.output_log, 'w', newline='') as writefile:  
                    writer = csv.writer(writefile)
                    writer.writerow([average_perplexity.item(), average_ner_perplexity.item(), average_noun_perplexity.item(), average_verb_perplexity.item(), average_num_perplexity.item()])
                
            elif 't5' in args.model_name_or_path:   
                for batch in iter(loader):
                    lm_labels = batch['target_ids']
                    targets = model.ids_to_clean_text(batch['target_ids'])
                    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                    outputs = model.model(
                        input_ids=batch["source_ids"].cuda(),
                        attention_mask=batch["source_mask"].cuda(),
                        labels=lm_labels.cuda(),
                        decoder_attention_mask=batch['target_mask'].cuda(),
                    )
                    loss = outputs[0]
                    total_loss += loss
                    batch_cnt += 1
                    print("perplexity", batch_cnt, loss.item(), torch.exp(total_loss/batch_cnt).item())
                with open(args.output_log, 'a', newline='') as writefile:  
                    writer = csv.writer(writefile)
                    writer.writerow([torch.exp(total_loss/batch_cnt).item()])
                print(f'Perplexity is: {torch.exp(total_loss/batch_cnt).item()}')