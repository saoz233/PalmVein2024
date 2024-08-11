import os
file_dir = './session1'
for i,file in enumerate(os.listdir(file_dir)):
    os.rename(os.path.join(file_dir, file), 
              os.path.join(file_dir, f'{i//10+1}_{(i)%10+1}.bmp'))