import argparse
import os
from utils import *

def mainAblation():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("GEMINI_KEY:")
    print(os.getenv("GEMINI_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    
        
    
    
    #### This is a promt by gpt
    #prompt = "First, identify the key concepts essential for answering the following question. Then, use these concepts to answer it.\n"
    
    #### This is a prompt by gemini
    longprompt = 'Carefully analyze the following question and deeply explore the core concepts, key terms, and potential underlying implications. Based on a thorough understanding of these concepts, consider all relevant factors and provide a comprehensive, accurate, and in-depth answer.\n'
    
    shortCoT = "Firstly find the key concepts which are crucial to answer the following question. Then answer the question considering the found concepts. Let's think step by step.\n"
    longCoT = "Carefully analyze the following question and deeply explore the core concepts, key terms, and potential underlying implications. Based on a thorough understanding of these concepts, consider all relevant factors and provide a comprehensive, accurate, and in-depth answer.  Let's think step by step.\n"
    
    #### This is a simple prompt of mine
    #prompt = "Firstly find the key concepts in the following question. Then answer the question based on these concepts.\n"
    
    #### This is my prompt
    #prompt1 = "Firstly find the key concepts which are crucial to answer the following question. Then answer the question considering the found concepts.\n"
    
    #### This is my step by step promt
    #prompt2 = "Answer he following question step by step. In every step, firstly find the key concepts to answer the question, and then finish the step considering the found concepts.\n"
    #x = prompt + "Q: " + x + "\n" + "A:"
    
    prompts = [longprompt, shortCoT, longCoT]
    accs = []
    for j in range(len(prompts)):
        total = 0
        correct_list = []
        if j >= 1:
            args.method = "zero_shot_cot"    
        for i, data in enumerate(dataloader):                    
            # Prepare question template ...
            x0, y = data
            
            #if x0.find('Christina is planning a birthday party and needs') < 0:
            #    continue
            x = prompts[j] + "Q: " + x0 + "\n" + "A:"
            #x = "Q: " + x + "\n" + "A:"
            
            #x = "Q: " + x[0] + "\n" + "A:"
            #y = y[0].strip()
            
            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            elif args.method == "few_shot":
                x = demo + x
            elif args.method == "few_shot_cot":
                x = demo + x
            else:
                raise ValueError("method is not properly defined ...")
            
            # Answer prediction by generating text ...
            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            z = decoder.decode(args, x, max_length, i, 1)
            #print( z )
            
    
            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot_cot":
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct
                pred = decoder.decode(args, z2, max_length, i, 2)
                #print(z2 + pred)
            else:
                pred = z
                
    
            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)
            
            # Choose the most frequent answer from the list ...
            '''print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')'''
            
            # Checking answer ...
            correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1 #np.array([y]).size(0)
            if correct == 0:
                print("{}st data".format(i+1))
                print("pred : {}".format(pred))
                print("GT : " + y)
                #print( x+'\n'+z )
            
            #if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            #    break
                #raise ValueError("Stop !!")
        
        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        accuracy = round(accuracy, 2)
        accs.append(accuracy)
        print(f"accuracy : {sum(correct_list)}\t{total}\t{accuracy}")
        print("accuracy : {}".format(accuracy))
    print( accs )    
    accsStr = f"{args.dataset}:\t\t{accs[0]}\t{accs[1]}\t{accs[2]}\n"
    with open("log.txt", 'at') as f:
        f.write(accsStr)

def mainMulti():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("GEMINI_KEY:")
    print(os.getenv("GEMINI_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    
        
    #prompt = "Firstly find the key concepts which are crucial to answer the following question. Then answer the question considering the found concepts. Remember to think step by step.\n"
    
    #### This is a promt by gpt
    #prompt = "First, identify the key concepts essential for answering the following question. Then, use these concepts to answer it.\n"
    
    #### This is a prompt by gemini
    #prompt = 'Carefully analyze the following question and deeply explore the core concepts, key terms, and potential underlying implications. Based on a thorough understanding of these concepts, consider all relevant factors and provide a comprehensive, accurate, and in-depth answer.\n'
    
    #### This is a simple prompt of mine
    #prompt = "Firstly find the key concepts in the following question. Then answer the question based on these concepts.\n"
    
    #### This is my prompt
    prompt1 = "Firstly find the key concepts which are crucial to answer the following question. Then answer the question considering the found concepts.\n"
    
    #### This is my step by step promt
    prompt2 = "Answer he following question step by step. In every step, firstly find the key concepts to answer the question, and then finish the step considering the found concepts.\n"
    #x = prompt + "Q: " + x + "\n" + "A:"
    
    prompts = ["", prompt1, "", prompt2]
    accs = []
    for j in range(len(prompts)):
        total = 0
        correct_list = []
        if j == 2:
            args.method = "zero_shot_cot"    
        for i, data in enumerate(dataloader):                    
            # Prepare question template ...
            x0, y = data
            
            #if x0.find('Christina is planning a birthday party and needs') < 0:
            #    continue
            x = prompts[j] + "Q: " + x0 + "\n" + "A:"
            #x = "Q: " + x + "\n" + "A:"
            
            #x = "Q: " + x[0] + "\n" + "A:"
            #y = y[0].strip()
            
            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            elif args.method == "few_shot":
                x = demo + x
            elif args.method == "few_shot_cot":
                x = demo + x
            else:
                raise ValueError("method is not properly defined ...")
            
            # Answer prediction by generating text ...
            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            z = decoder.decode(args, x, max_length, i, 1)
            #print( z )
            
    
            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot_cot":
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct
                pred = decoder.decode(args, z2, max_length, i, 2)
                #print(z2 + pred)
            else:
                pred = z
                
    
            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)
            
            # Choose the most frequent answer from the list ...
            '''print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')'''
            
            # Checking answer ...
            correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1 #np.array([y]).size(0)
            if correct == 0:
                print("{}st data".format(i+1))
                print("pred : {}".format(pred))
                print("GT : " + y)
                #print( x+'\n'+z )
            
            #if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            #    break
                #raise ValueError("Stop !!")
        
        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        accuracy = round(accuracy, 2)
        accs.append(accuracy)
        print(f"accuracy : {sum(correct_list)}\t{total}\t{accuracy}")
        print("accuracy : {}".format(accuracy))
    accsStr = f"{args.dataset}:\t\t{accs[0]}\t{accs[1]}\t{accs[2]}\t{accs[3]}\n"
    with open("log.txt", 'at') as f:
        f.write(accsStr)
    
def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("GEMINI_KEY:")
    print(os.getenv("GEMINI_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    
    total = 0
    correct_list = []    
    single = -1    
    for i, data in enumerate(dataloader):
        if single >= 0:
            if i != single:
                continue
        
        #print('*************************')
        #print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        #prompt = "Firstly find the key concepts which are crucial to answer the following question. Then answer the question considering the found concepts. Remember to think step by step.\n"
        
        #### This is a promt by gpt
        #prompt = "First, identify the key concepts essential for answering the following question. Then, use these concepts to answer it.\n"
        
        #### This is a prompt by gemini
        #prompt = 'Carefully analyze the following question and deeply explore the core concepts, key terms, and potential underlying implications. Based on a thorough understanding of these concepts, consider all relevant factors and provide a comprehensive, accurate, and in-depth answer.\n'
        
        #### This is a simple prompt of mine
        #prompt = "Firstly find the key concepts in the following question. Then answer the question based on these concepts.\n"
        
        #### This is my prompt
        #prompt = "Firstly find the key concepts which are crucial to answer the following question. Then answer the question considering the found concepts.\n"
        
        #### This is my step by step promt
        #prompt = "Answer he following question step by step. In every step, firstly find the key concepts to answer the question, and then finish the step considering the found concepts.\n"
        #x = prompt + "Q: " + x + "\n" + "A:"
        
        x = "Q: " + x + "\n" + "A:"
        
        #x = "Q: " + x[0] + "\n" + "A:"
        #y = y[0].strip()
        
        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x
        else:
            raise ValueError("method is not properly defined ...")
        
        # Answer prediction by generating text ...
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        z = decoder.decode(args, x, max_length, i, 1)
        #print( z )
        if single >= 0:
            print(x + z)

        # Answer extraction for zero-shot-cot ...
        if args.method == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)
            #print(z2 + pred)
        else:
            pred = z
            

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        
        # Choose the most frequent answer from the list ...
        '''print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')'''
        
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        if correct == 0:
            print("{}st data".format(i+1))
            print("pred : {}".format(pred))
            print("GT : " + y)
            #print( x+'\n'+z )
        
        #if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
        #    break
            #raise ValueError("Stop !!")
    
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    ''' Chinese_Anachronisms_Judgment --- 150
    Chinese_Movie_and_Music_Recommendation --- 50
    Chinese_Natural_Language_Inference --- 100
    Chinese_Reading_Comprehension --- 200
    Chinese_Sequence_Understanding --- 100
    Chinese_Sport_Understanding --- 200
    Chinese_Time_Understanding --- 100
    Global_Anachronisms_Judgment --- 150
    Global_Movie_and_Music_Recommendation --- 50
    Global_Natural_Language_Inference --- 100
    Global_Reading_Comprehension --- 200
    Global_Sequence_Understanding --- 100
    Global_Sport_Understanding --- 200
    Global_Time_Understanding --- 100'''
    
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gemini-1.5-flash", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=20.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset.startswith( 'Chinese' ) or args.dataset.startswith( 'Global' ):
        args.dataset_path = f"./dataset/reasoning/{args.dataset}.json"
        if args.dataset.count('Anachronisms')>0 or args.dataset.count('Sport')>0 :
            args.direct_answer_trigger = "\nTherefore, among A through B, the answer is"
        elif args.dataset.count('Natural')>0 :
            args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
        else:
            args.direct_answer_trigger = "\nTherefore, among A through D, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return args

if __name__ == "__main__":
    mainAblation()