import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import torch
import pickle
from tqdm import tqdm
import re
import time
from torch.utils.data import DataLoader, Dataset
import gpustat
from tqdm import tqdm
import time
from helpers.llm_helper import *
from helpers.llm_prompts import *


def main():
    ##### Load Data ##########
    data_folder = '/radraid2/dongwoolee/RadPath/data' # change according to your system
    radreport_path = os.path.join(data_folder, 'mrnacc_ultrasound_generate_radreport.pkl')
    with open(radreport_path, 'rb') as f:
        all_radreport = pickle.load(f)

    bxreport_path = os.path.join(data_folder, 'mrnacc_ultrasound_generate_bxreport.pkl')
    with open(bxreport_path, 'rb') as f:
        all_bxreport = pickle.load(f)

    # Set output file for JSON
    output_file = str(input("Enter file name to save the data (end with .json, default='inference_results.json'): "))
    if not output_file:
        output_file = "inference_results.json"

    ##### Load Model #########
    tokenizer, model = load_model(
        cache_dir="/radraid2/dongwoolee/.llms", # change according to your system
        model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        #model_name="unsloth/QwQ-32B-unsloth-bnb-4bit"
        )

    ###########################################################################
    #                                Put MRN Here
    ###########################################################################
    dev_mrns = ['mrn839-acc839', 'mrn993-acc993', 'mrn1013-acc1013', 'mrn627-acc627', 'mrn241-acc241', 
            'mrn674-acc674', 'mrn189-acc189', 'mrn34-acc34', 'mrn1070-acc1070', 'mrn4-acc4', 
            'mrn489-acc489', 'mrn1006-acc1006', 'mrn671-acc671', 'mrn520-acc520', 'mrn233-acc233', 
            'mrn282-acc282', 'mrn32-acc32', 'mrn676-acc676', 'mrn230-acc230', 'mrn727-acc727', 
            'mrn726-acc726', 'mrn719-acc719', 'mrn826-acc826', 'mrn654-acc654', 'mrn524-acc524',
            'mrn995-acc995', 'mrn376-acc376', 'mrn957-acc957', 'mrn747-acc747', 'mrn251-acc251', 
            'mrn498-acc498', 'mrn276-acc276', 'mrn768-acc768', 'mrn742-acc742', 'mrn392-acc392', 
            'mrn1014-acc1014', 'mrn258-acc258', 'mrn502-acc502', 'mrn466-acc466', 'mrn1023-acc1023', 
            'mrn546-acc546', 'mrn1058-acc1058', 'mrn663-acc663', 'mrn645-acc645', 'mrn804-acc804', 
            'mrn315-acc315', 'mrn85-acc85', 'mrn724-acc724', 'mrn35-acc35', 'mrn418-acc418',
            'mrn1052-acc1052', 'mrn769-acc769', 'mrn683-acc683', 'mrn686-acc686', 'mrn929-acc929', 
            'mrn863-acc863', 'mrn812-acc812', 'mrn206-acc206', 'mrn648-acc648', 'mrn729-acc729', 
            'mrn1021-acc1021', 'mrn1001-acc1001', 'mrn748-acc748', 'mrn556-acc556', 'mrn1053-acc1053', 
            'mrn612-acc612', 'mrn1050-acc1050', 'mrn1072-acc1072', 'mrn702-acc702', 'mrn459-acc459', 
            'mrn962-acc962', 'mrn709-acc709', 'mrn130-acc130', 'mrn870-acc870', 'mrn975-acc975',
            'mrn555-acc555', 'mrn776-acc776', 'mrn52-acc52', 'mrn14-acc14', 'mrn871-acc871', 
            'mrn577-acc577', 'mrn358-acc358', 'mrn1025-acc1025', 'mrn242-acc242', 'mrn972-acc972', 
            'mrn134-acc134', 'mrn738-acc738', 'mrn921-acc921', 'mrn794-acc794', 'mrn406-acc406', 
            'mrn592-acc592', 'mrn838-acc838', 'mrn410-acc410', 'mrn805-acc805', 'mrn795-acc795', 
            'mrn897-acc897', 'mrn339-acc339', 'mrn379-acc379', 'mrn252-acc252', 'mrn62-acc62']
    test_mrns = ['mrn1-acc1', 'mrn2-acc2', 'mrn3-acc3', 'mrn6-acc6', 'mrn8-acc8', 'mrn11-acc11', 
                 'mrn12-acc12', 'mrn16-acc16', 'mrn17-acc17', 'mrn18-acc18', 'mrn22-acc22', 'mrn24-acc24', 
                 'mrn26-acc26', 'mrn29-acc29', 'mrn30-acc30', 'mrn33-acc33', 'mrn45-acc45', 'mrn48-acc48', 
                 'mrn49-acc49', 'mrn55-acc55', 'mrn60-acc60', 'mrn63-acc63', 'mrn65-acc65', 'mrn66-acc66', 
                 'mrn72-acc72', 'mrn75-acc75', 'mrn76-acc76', 'mrn77-acc77', 'mrn82-acc82', 'mrn83-acc83', 
                 'mrn92-acc92', 'mrn94-acc94', 'mrn97-acc97', 'mrn100-acc100', 'mrn109-acc109', 
                 'mrn111-acc111', 'mrn118-acc118', 'mrn120-acc120', 'mrn125-acc125', 'mrn131-acc131', 'mrn132-acc132', 
                 'mrn133-acc133', 'mrn135-acc135', 'mrn137-acc137', 'mrn142-acc142', 'mrn146-acc146', 'mrn147-acc147', 
                 'mrn148-acc148', 'mrn149-acc149', 'mrn153-acc153', 'mrn154-acc154', 'mrn157-acc157', 'mrn158-acc158', 
                 'mrn160-acc160', 'mrn162-acc162', 'mrn163-acc163', 'mrn168-acc168', 'mrn171-acc171', 'mrn178-acc178', 
                 'mrn179-acc179', 'mrn181-acc181', 'mrn188-acc188', 'mrn190-acc190', 'mrn198-acc198', 'mrn199-acc199', 
                 'mrn203-acc203', 'mrn204-acc204', 'mrn207-acc207', 'mrn209-acc209', 'mrn210-acc210', 'mrn211-acc211', 
                 'mrn223-acc223', 'mrn225-acc225', 'mrn226-acc226', 'mrn227-acc227', 'mrn228-acc228', 'mrn234-acc234', 
                 'mrn239-acc239', 'mrn243-acc243', 'mrn245-acc245', 'mrn250-acc250', 'mrn253-acc253', 'mrn256-acc256', 
                 'mrn259-acc259', 'mrn262-acc262', 'mrn264-acc264', 'mrn274-acc274', 'mrn277-acc277', 'mrn278-acc278', 
                 'mrn279-acc279', 'mrn280-acc280', 'mrn281-acc281', 'mrn284-acc284', 'mrn286-acc286', 'mrn291-acc291', 
                 'mrn298-acc298', 'mrn299-acc299', 'mrn301-acc301', 'mrn302-acc302', 'mrn303-acc303', 'mrn304-acc304', 
                 'mrn309-acc309', 'mrn311-acc311', 'mrn328-acc328', 'mrn331-acc331', 'mrn332-acc332', 'mrn342-acc342', 
                 'mrn344-acc344', 'mrn348-acc348', 'mrn349-acc349', 'mrn350-acc350', 'mrn351-acc351', 'mrn352-acc352', 
                 'mrn354-acc354', 'mrn361-acc361', 'mrn365-acc365', 'mrn369-acc369', 'mrn372-acc372', 'mrn374-acc374', 
                 'mrn375-acc375', 'mrn377-acc377', 'mrn382-acc382', 'mrn383-acc383', 'mrn386-acc386', 'mrn388-acc388', 
                 'mrn390-acc390', 'mrn393-acc393', 'mrn395-acc395', 'mrn404-acc404', 'mrn407-acc407', 'mrn408-acc408', 
                 'mrn411-acc411', 'mrn412-acc412', 'mrn414-acc414', 'mrn416-acc416', 'mrn419-acc419', 'mrn424-acc424', 
                 'mrn426-acc426', 'mrn427-acc427', 'mrn429-acc429', 'mrn431-acc431', 'mrn434-acc434', 'mrn440-acc440', 
                 'mrn447-acc447', 'mrn448-acc448', 'mrn452-acc452', 'mrn454-acc454', 'mrn462-acc462', 'mrn468-acc468', 
                 'mrn469-acc469', 'mrn473-acc473', 'mrn475-acc475', 'mrn476-acc476', 'mrn478-acc478', 'mrn482-acc482', 
                 'mrn485-acc485', 'mrn487-acc487', 'mrn488-acc488', 'mrn490-acc490', 'mrn491-acc491', 'mrn494-acc494', 
                 'mrn495-acc495', 'mrn499-acc499', 'mrn500-acc500', 'mrn501-acc501', 'mrn511-acc511', 'mrn521-acc521', 
                 'mrn522-acc522', 'mrn523-acc523', 'mrn525-acc525', 'mrn529-acc529', 'mrn531-acc531', 'mrn535-acc535', 
                 'mrn536-acc536', 'mrn539-acc539', 'mrn542-acc542', 'mrn543-acc543', 'mrn545-acc545', 'mrn548-acc548', 
                 'mrn549-acc549', 'mrn557-acc557', 'mrn567-acc567', 'mrn570-acc570', 'mrn572-acc572', 'mrn576-acc576', 
                 'mrn578-acc578', 'mrn579-acc579', 'mrn580-acc580', 'mrn581-acc581', 'mrn583-acc583', 'mrn585-acc585', 
                 'mrn587-acc587', 'mrn590-acc590', 'mrn595-acc595', 'mrn599-acc599', 'mrn601-acc601', 'mrn603-acc603', 
                 'mrn604-acc604', 'mrn606-acc606', 'mrn608-acc608', 'mrn610-acc610', 'mrn614-acc614', 'mrn615-acc615', 
                 'mrn616-acc616', 'mrn617-acc617', 'mrn623-acc623', 'mrn636-acc636', 'mrn650-acc650', 'mrn656-acc656', 
                 'mrn657-acc657', 'mrn658-acc658', 'mrn660-acc660', 'mrn661-acc661', 'mrn666-acc666', 'mrn668-acc668', 
                 'mrn670-acc670', 'mrn672-acc672', 'mrn680-acc680', 'mrn684-acc684', 'mrn692-acc692', 'mrn693-acc693', 
                 'mrn695-acc695', 'mrn696-acc696', 'mrn703-acc703', 'mrn704-acc704', 'mrn705-acc705', 'mrn708-acc708', 
                 'mrn713-acc713', 'mrn718-acc718', 'mrn723-acc723', 'mrn730-acc730', 'mrn731-acc731', 'mrn734-acc734', 
                 'mrn735-acc735', 'mrn739-acc739', 'mrn740-acc740', 'mrn745-acc745', 'mrn749-acc749', 'mrn750-acc750', 
                 'mrn751-acc751', 'mrn752-acc752', 'mrn755-acc755', 'mrn756-acc756', 'mrn757-acc757', 'mrn761-acc761', 
                 'mrn762-acc762', 'mrn765-acc765', 'mrn767-acc767', 'mrn770-acc770', 'mrn781-acc781', 'mrn783-acc783', 
                 'mrn785-acc785', 'mrn787-acc787', 'mrn796-acc796', 'mrn803-acc803', 'mrn806-acc806', 'mrn809-acc809', 
                 'mrn813-acc813', 'mrn816-acc816', 'mrn820-acc820', 'mrn821-acc821', 'mrn822-acc822', 'mrn823-acc823', 
                 'mrn824-acc824', 'mrn825-acc825', 'mrn827-acc827', 'mrn828-acc828', 'mrn830-acc830', 'mrn832-acc832', 
                 'mrn834-acc834', 'mrn842-acc842', 'mrn843-acc843', 'mrn845-acc845', 'mrn846-acc846', 'mrn848-acc848', 
                 'mrn849-acc849', 'mrn854-acc854', 'mrn858-acc858', 'mrn864-acc864', 'mrn866-acc866', 'mrn872-acc872', 
                 'mrn873-acc873', 'mrn875-acc875', 'mrn877-acc877', 'mrn879-acc879', 'mrn881-acc881', 'mrn883-acc883', 
                 'mrn884-acc884', 'mrn887-acc887', 'mrn892-acc892', 'mrn893-acc893', 'mrn894-acc894', 'mrn896-acc896', 
                 'mrn901-acc901', 'mrn904-acc904', 'mrn905-acc905', 'mrn910-acc910', 'mrn914-acc914', 'mrn916-acc916', 
                 'mrn919-acc919', 'mrn920-acc920', 'mrn923-acc923', 'mrn924-acc924', 'mrn938-acc938', 'mrn940-acc940', 
                 'mrn941-acc941', 'mrn948-acc948', 'mrn952-acc952', 'mrn953-acc953', 'mrn958-acc958', 'mrn959-acc959', 
                 'mrn963-acc963', 'mrn964-acc964', 'mrn965-acc965', 'mrn967-acc967', 'mrn968-acc968', 'mrn969-acc969', 
                 'mrn973-acc973', 'mrn974-acc974', 'mrn976-acc976', 'mrn977-acc977', 'mrn983-acc983', 'mrn985-acc985', 
                 'mrn987-acc987', 'mrn990-acc990', 'mrn991-acc991', 'mrn992-acc992', 'mrn998-acc998', 'mrn999-acc999', 
                 'mrn1000-acc1000', 'mrn1003-acc1003', 'mrn1004-acc1004', 'mrn1005-acc1005', 'mrn1008-acc1008', 'mrn1010-acc1010', 
                 'mrn1012-acc1012', 'mrn1016-acc1016', 'mrn1017-acc1017', 'mrn1018-acc1018', 'mrn1022-acc1022', 'mrn1029-acc1029', 
                 'mrn1031-acc1031', 'mrn1034-acc1034', 'mrn1036-acc1036', 'mrn1039-acc1039', 'mrn1042-acc1042', 'mrn1043-acc1043', 
                 'mrn1047-acc1047', 'mrn1048-acc1048', 'mrn1060-acc1060', 'mrn1074-acc1074']

    steven_new50 = ['mrn55-acc55', 'mrn72-acc72', 'mrn75-acc75', 'mrn111-acc111', 'mrn132-acc132', 'mrn149-acc149', 
                    'mrn188-acc188', 'mrn204-acc204', 'mrn209-acc209', 'mrn227-acc227', 'mrn256-acc256', 'mrn274-acc274', 
                    'mrn277-acc277', 'mrn301-acc301', 'mrn309-acc309', 'mrn383-acc383', 'mrn393-acc393', 'mrn407-acc407', 
                    'mrn408-acc408', 'mrn431-acc431', 'mrn469-acc469', 'mrn494-acc494', 'mrn495-acc495', 'mrn535-acc535', 
                    'mrn572-acc572', 'mrn576-acc576', 'mrn590-acc590', 'mrn616-acc616', 'mrn650-acc650', 'mrn723-acc723', 
                    'mrn730-acc730', 'mrn739-acc739', 'mrn749-acc749', 'mrn756-acc756', 'mrn767-acc767', 'mrn803-acc803', 
                    'mrn828-acc828', 'mrn832-acc832', 'mrn845-acc845', 'mrn854-acc854', 'mrn875-acc875', 'mrn923-acc923', 
                    'mrn959-acc959', 'mrn974-acc974', 'mrn990-acc990', 'mrn998-acc998', 'mrn1003-acc1003', 'mrn1008-acc1008', 
                    'mrn1017-acc1017', 'mrn1039-acc1039']
    
    bx_loc_error_mrns = ["mrn6-acc6", "mrn132-acc132", "mrn239-acc239", "mrn303-acc303", "mrn365-acc365", "mrn390-acc390", "mrn395-acc395",
                         "mrn434-acc434", "mrn447-acc447", "mrn452-acc452", "mrn462-acc462", "mrn485-acc485", "mrn522-acc522", "mrn535-acc535",
                         "mrn739-acc739", "mrn834-acc834", "mrn877-acc877", "mrn896-acc896", "mrn901-acc901", "mrn916-acc916",  "mrn974-acc974"]
    
    chandler_old50 = ['mrn2-acc2', 'mrn8-acc8', 'mrn12-acc12', 'mrn16-acc16', 'mrn24-acc24', 'mrn48-acc48', 
                      'mrn49-acc49', 'mrn63-acc63', 'mrn65-acc65', 'mrn66-acc66', 'mrn76-acc76', 'mrn92-acc92', 
                      'mrn94-acc94', 'mrn97-acc97', 'mrn100-acc100', 'mrn125-acc125', 'mrn131-acc131', 'mrn133-acc133', 
                      'mrn137-acc137', 'mrn142-acc142', 'mrn147-acc147', 'mrn148-acc148', 'mrn158-acc158', 'mrn160-acc160', 
                      'mrn163-acc163', 'mrn178-acc178', 'mrn190-acc190', 'mrn198-acc198', 'mrn199-acc199', 'mrn203-acc203', 
                      'mrn211-acc211', 'mrn223-acc223', 'mrn225-acc225', 'mrn228-acc228', 'mrn253-acc253', 'mrn264-acc264', 
                      'mrn279-acc279', 'mrn284-acc284', 'mrn299-acc299', 'mrn302-acc302', 'mrn304-acc304', 'mrn344-acc344', 
                      'mrn352-acc352', 'mrn382-acc382', 'mrn388-acc388', 'mrn411-acc411', 'mrn412-acc412', 'mrn416-acc416', 'mrn426-acc426', 'mrn468-acc468']
    
    chandler_new50 = ['mrn473-acc473', 'mrn475-acc475', 'mrn476-acc476', 'mrn478-acc478', 'mrn491-acc491', 'mrn521-acc521', 
                      'mrn531-acc531', 'mrn536-acc536', 'mrn539-acc539', 'mrn542-acc542', 'mrn543-acc543', 'mrn545-acc545', 
                      'mrn548-acc548', 'mrn549-acc549', 'mrn557-acc557', 'mrn578-acc578', 'mrn587-acc587', 'mrn599-acc599', 
                      'mrn606-acc606', 'mrn608-acc608', 'mrn615-acc615', 'mrn623-acc623', 'mrn666-acc666', 'mrn680-acc680', 
                      'mrn731-acc731', 'mrn751-acc751', 'mrn761-acc761', 'mrn765-acc765', 'mrn783-acc783', 'mrn806-acc806', 
                      'mrn813-acc813', 'mrn842-acc842', 'mrn858-acc858', 'mrn887-acc887', 'mrn893-acc893', 'mrn904-acc904', 
                      'mrn905-acc905', 'mrn919-acc919', 'mrn924-acc924', 'mrn938-acc938', 'mrn940-acc940', 'mrn948-acc948', 
                      'mrn953-acc953', 'mrn967-acc967', 'mrn968-acc968', 'mrn973-acc973', 'mrn991-acc991', 'mrn1004-acc1004', 'mrn1031-acc1031', 'mrn1043-acc1043']
    Qwen_match_err = ['mrn60-acc60', 'mrn82-acc82', 'mrn83-acc83', 'mrn211-acc211', 'mrn286-acc286', 'mrn390-acc390', 'mrn407-acc407', 
                      'mrn408-acc408', 'mrn434-acc434', 'mrn482-acc482', 'mrn535-acc535', 'mrn834-acc834',  'mrn919-acc919']

    mrns = test_mrns
    print(mrns)
    ############################################################################
    ####### Refine MRNs #######################################################
    output_file = os.path.join(os.getcwd(), "llm_results", output_file)
    elapsed_time_list = []
    # Load previous results (if any)
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                results = json.load(f)  # Load existing results
        except (json.JSONDecodeError, IOError):
            print("Warning: Corrupt or empty JSON file. Resetting results.")
            results = {}  # Reset if file is corrupted
    else:
        results = {}

    processed_mrns = set(results.keys())    
    mrns = [mrn for mrn in mrns if mrn not in processed_mrns]
    print(f"Remaining MRNs to process: {len(mrns)}") 
   
    ###### Create Dataset and Define DataLoader ####################################
    dataset = PairedReportDataset(all_radreport, all_bxreport, mrns)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    ###### Process Each Batch and Append Results ############################
    pbar = tqdm(dataloader, desc="Processing Reports")
    for batch in pbar:
        # Unpack the batch (each batch contains one sample since batch_size=1)
        mrns_batch, radreports_batch, bxreports_batch = batch
        mrns_batch = list(mrns_batch)
        radreports_batch = list(radreports_batch)
        bxreports_batch = list(bxreports_batch)

        batch_start_time = time.strftime("%H:%M:%S")
        start_time = time.time()
        pbar.set_postfix({"batch_start": batch_start_time, "MRN": mrns_batch})

        batch_messages = prompt_chat_template_Qwen1(bxreports_batch, radreports_batch)
        batch_generated_texts = generate_text_with_chat_template(tokenizer, model, batch_messages, do_sample=False, temperature=None, top_p=None, max_tokens=32768)
        #batch_generated_texts = generate_text_with_chat_template(tokenizer, model, batch_messages, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768) # Qwen
        
        elapsed_time = time.time()-start_time
        elapsed_time_list.append(elapsed_time)
        for mrn, gen_text in zip(mrns_batch, batch_generated_texts):
            results[mrn] = gen_text

        with open(output_file, "w") as outfile:
            json.dump(results, outfile, indent=4)
            outfile.flush()
        
        with open(f"{output_file.strip('.json')}_elapsed_time.json", "w") as f:
            json.dump(elapsed_time_list, f)
            f.flush()

    print("Inference completed. Results appended in 'inference_results.json'.")

if __name__=='__main__':
    main()