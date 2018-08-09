import matplotlib
matplotlib.use('TkAgg')
from tkinter import *
#import tkMessageBox
#import tkFileDialog
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog
import cv2

from MVCCloner import MVCCloner
from MVCCloner_removal import MVCCloner_removal

varrs = []

loop_count = 0

hold_target = None

check = 1

def select_source_image(A):

    global src_img_path
    src_img_path = tkFileDialog.askopenfilename(title='Please select a src to analyze',filetypes=[('Jpg files','*.jpg'), ('Jpeg file','*.jpeg'), ('Png file', '*.png')])
    if len(src_img_path)>0:
      A.select()
   

def select_target_image(AA):
   
    global target_img_path
    target_img_path = tkFileDialog.askopenfilename(title='Please select a target to analyze',filetypes=[('Jpg files','*.jpg'), ('Jpeg file','*.jpeg'), ('Png file', '*.png')])
    global check
    check = 0
    if len(target_img_path)>0:
      AA.select()
    

def compute_image_cloning(varrs):
    mvc_config = {'hierarchic': True,
                'base_angle_Th': 0.75,
                'base_angle_exp': 0.8,
                'base_length_Th': 2.5,
                'adaptiveMeshShapeCriteria': 0.125,
                'adaptiveMeshSizeCriteria': 0.,
                'min_h_res': 16.}

    global target_img_path
    global src_img_path
    global loop_count
    global hold_target
    global check

    if src_img_path and target_img_path:

        if not varrs[0].get() and not varrs[1].get():
          tkMessageBox.showinfo('Input Error','Please select Cloning Type')
        else:
          if varrs[0].get():
          
            if loop_count != 0 and check:
                target_img_path = './out.jpg'              
            mvc_cloner = MVCCloner(src_img_path, target_img_path, './out.jpg', mvc_config,varrs[3].get())
            mvc_cloner.GetPatch()
            mvc_cloner.run()
            loop_count = loop_count + 1
            check = 1

            hold_target = target_img_path
          if varrs[1].get():
           
            if loop_count != 0 and check:
                src_img_path = './out.jpg'
                target_img_path = './out.jpg'
            mvc_cloner = MVCCloner_removal(src_img_path, src_img_path, './out_removal.jpg', mvc_config,varrs[3].get())
            mvc_cloner.GetPatch()
            mvc_cloner.run()
            loop_count = loop_count + 1
            check = 1
    else:
      tkMessageBox.showinfo('Input Error','Please select input images')
      
      #tkMessageBox.askyesno("Title", "Your question goes here?")


def runSelectedItems(varrs):

    print(len(varrs))

    for idx, i in enumerate(varrs):
      if i.get() == 1:
        print(idx)
    #if checkCmd == 0:
    #    labelText = Label(text="It worked").pack()
    #else:
    #    labelText = Label(text="Please select an item from the checklist below").pack()



if __name__ == '__main__':
    ## argument parse ##
    #parser = argparse.ArgumentParser(description="Mean-Value Seamless Cloning")
    #parser.add_argument("src", help="The path to source image")
    #parser.add_argument("target", help="The path to target image")
    #parser.add_argument("-o", "--output", help="The path to save the cloning image", default="./out.jpg")
    #args = parser.parse_args()

    # set arguments #
    #src_img_path = args.src
    #target_img_path = args.target
    #output_path = args.output
    src_img_path = None
    target_img_path = None

    root = Tk()
    root.geometry("550x300+300+150")
    root.resizable(width=True, height=True)
    panelA = None
    panelB = None

    var = IntVar()
    A = Checkbutton(root, text="Src Done", variable=var)

    var2 = IntVar()
    AA = Checkbutton(root, text="Target Done", variable=var2)

    btn = Button(root, text="Select source image", command= lambda: select_source_image(A),height =2, width = 30)

    btn.pack()
    A.pack()


    btn2 = Button(root, text="Select target image",command=lambda: select_target_image(AA),height =2, width = 30)
    btn2.pack()
    AA.pack()

    btn3 = Button(root, text="Compute Image Cloning",command=lambda: compute_image_cloning(varrs),  width = 30)
    btn3.pack()

    picks = ["MVC Blending","Object Removal","Poisson Blending", "Selective Edge"]
    for pick in picks:
        var = IntVar()
        chk = Checkbutton(root, text=pick, variable=var)
        chk.pack(side = LEFT,anchor=W, expand=YES)
        varrs.append(var)
    cv2.destroyAllWindows()    

    #for i in varrs:
    #  i.pack(side=LEFT, anchor=W, expand=YES)


    root.mainloop()
