import cv2
import keyboard
import os

def save_image(type_file, sample, n_samples_collected, frame,n_images):
    if n_images>(0.7*n_images_collect):
        os.chdir(current_dir + '\\' +'Dataset'+'\\'+dataset_type[1]+'\\'+ file_name[type_file])
    else:
        os.chdir(current_dir + '\\' +'Dataset'+'\\'+dataset_type[0]+'\\'+ file_name[type_file])
    cv2.imwrite(f'{file_name[type_file]}{n_samples_collected}.png', frame[105:395, 285:595])
    n_samples_collected += 1
    n_images +=1
    if n_images == n_images_collect:
        sample = False
        n_images=0
    return sample, n_samples_collected,n_images


# creating files to store the images in

current_dir=os.getcwd()

dataset_type=['train','validation']
file_name=['click','move','dummy']
try:
    os.mkdir(current_dir + '\\' + 'Dataset')
    for name in dataset_type:
        os.mkdir(current_dir+'\\'+'Dataset'+'\\'+name)
        for i in file_name:
            os.mkdir(current_dir + '\\' +'Dataset'+'\\'+name+'\\'+ i)
except:
    pass

n_images_collect=5000
Click, Move,Dummy= [False]*3
n_Click, n_Move,n_Dummy, n_images= [len(os.listdir(current_dir + '\\' +'Dataset'+'\\'+dataset_type[0]+'\\'+ i))+len(os.listdir(current_dir + '\\' +'Dataset'+'\\'+dataset_type[1]+'\\'+ i)) for i in file_name]+[0]

vid=cv2.VideoCapture(0)
while True:
    success, frame=vid.read()
    if not success:
        print('ERROR')
        break

    frame=cv2.flip(frame,1)
    cv2.putText(frame, 'Click: ' + str(n_Click), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'Move: ' + str(n_Move), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'Dummy: '+str(n_Dummy), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame,(280,100),(600,400),(0,0,255),2)
    cv2.imshow('video', frame)

    if keyboard.is_pressed('c'):
        if (Move,Dummy)==(False,False):
            Click=True
            print('collecting Clicks')
        else:
            print('unavailable right now')

    if Click:
        Click, n_Click, n_images=save_image(0, Click, n_Click, frame, n_images)


    if keyboard.is_pressed('m'):
        if (Click,Dummy)==(False, False):
            Move=True
            print('collecting move')
        else:
            print('unavailable right now')

    if Move:
        Move, n_Move, n_images=save_image(1, Move, n_Move, frame, n_images)

    if keyboard.is_pressed('d'):
        if (Click, Move) == (False, False):
            Dummy = True
            print('collecting dummy')
        else:
            print('unavailable right now')

    if Dummy:
        Dummy, n_Dummy, n_images = save_image(2, Dummy, n_Dummy, frame, n_images)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
