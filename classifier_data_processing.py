

#load stag1 model
test_m = torch.load("/content/drive/My Drive/Project/checkpoint2/Stg2AE_5100")
test_m.eval()

for param in test_m.parameters():
    param.requires_grad = False

stg2_en = test_m.En
stg2_de = test_m.De

#load stag2 model
stg2_m = torch.load("/content/drive/My Drive/Project/checkpoint2/Stg2AE_5100")
stg2_m.eval()

for param in stg2_m.parameters():
    param.requires_grad = False

stg2_en = stg2_m.En
stg2_de = stg2_m.De

#load stag3 model
stg3_m = torch.load("/content/drive/My Drive/Project/checkpoint2/Stg2AE_5100")
stg3_m.eval()

for param in stg3_m.parameters():
    param.requires_grad = False

stg2_en = stg3_m.En
stg2_de = stg3_m.De

############################# convert all the images to stage 3 features ######################

transform= transforms.Compose([transforms.ToTensor()])
hf = h5py.File("/content/drive/My Drive/Project/faceScrub/img.h5", "r")
stg1_dataset = h5py.File("/content/drive/My Drive/Project/faceScrub/img_stg3_feat3_class.h5", "w")
stg1_feature = stg1_dataset.create_dataset("stg3_feature3", [8000, 1000])
stg1_names = stg1_dataset.create_dataset("names",(8000,), dtype=h5py.string_dtype())


for id in range(8000):

    stg1_dataset["names"][id,...]=hf["names"][id,...]
    data=hf['train_img'][id,...]
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = Image.fromarray(data)
    
    data = transform(data)
    data = torch.Tensor(data)
    data = data.to(device)

    out = stg3_en(stg2_en(stg1_en(data.unsqueeze(0)))) #stg1_en read from above
    stg1_dataset["stg3_feature3"][id] = out.squeeze().cpu().detach().numpy()
    
hf.close()
stg1_dataset.close()


############################# pair stage 3 features to form calssifier input ######################
import random
hf_new = h5py.File('/content/drive/My Drive/Project/faceScrub/520_img.h5','r+')
stg1_dataset =  h5py.File("/content/drive/My Drive/Project/faceScrub/520_img_stg3_feat3_class.h5", "r")
hf_paired = h5py.File("/content/drive/My Drive/Project/faceScrub/520_classifier_input2.h5", "r+")
name = hf_paired.create_dataset("names",(16000,), dtype=h5py.string_dtype())
data = hf_paired.create_dataset("data",[16000,2000])

new_i=0
for i in range(400):
	for k in range(20):
		j=i*20
		rand= random.randint(0,19)
		hf_paired["data"][new_i] = torch.cat((torch.tensor(stg1_dataset["stg3_feature3"][j+k]), torch.tensor(stg1_dataset["stg3_feature3"][j+rand])), 0)
		hf_paired["names"][new_i,...]=int(hf_new["names"][j+k,...].split()[0]==hf_new["names"][j+rand,...].split()[0])
		new_i+=1


	
for i in range(400):
	for k in range(20):
		j=i*20
		rand= random.randint(0,8000)
		hf_paired["data"][new_i] = torch.cat((torch.tensor(stg1_dataset["stg3_feature3"][j+k]), torch.tensor(stg1_dataset["stg3_feature3"][rand])), 0)     
		hf_paired["names"][new_i,...]=int(hf_new["names"][j+k,...].split()[0]==hf_new["names"][rand,...].split()[0])
		new_i+=1
				
hf_paired.close()		
stg1_dataset.close()	
hf_new.close()	