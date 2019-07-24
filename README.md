Arabic Handwritten Text Detection and Recognization




1.	Dataset used:-


    •	http://khatt.ideas2serve.net/KHATTAgreement.php
    
    •	We can use line images for training the model
    
    •	We can use Ground Truth Unicode Truth Values xlsx files

2.	Code


    •	Base Code is same as the one used by Antworks Bangalore team for recognizing English handwritten text.
    
    •	But the problem with that was it can only detect 32 characters from image. But our dataset has line text images so it has around 100 characters.
      
      So I used more cnn layers from 5 to 7.
      
      And I also made some other changes to code for using 7 layers
      
      https://towardsdatascience.com/faq-build-a-handwritten-text-recognition-system-using-tensorflow-27648fb18519
      
    •	Since dataset used is also different so I had to make changes in DataLoader.py file
    
      Like in Sample(atext, fileName) 
      
      atext is Arabic text taken from ground truth excel file
      
      fileName is name of the images used.
      
    •	Also encoding should be "utf-8" since Arabic characters cannot be in “ANSI” or default encoding.
    
    •	Since terminal can’t show Arabic characters, we have to write it to a file.
    

3.	Conclusions


    •	Accuracy is low 
    
      Character error rate is 61%
      
      Word error rate very high 99%
      
    •	Accuracy Is low because we are using line images instead of word
    
      Images are not cropped properly or was skewed.
      
    •	If we can crop properly and aline images properly, we can improve the accuracy
    
    •	Or we can find new dataset to train upon
    


4.	Refrences 


    •	https://github.com/githubharald/SimpleHTR
    
    •	https://medium.com/m/global-identity?redirectUrl=https%3A%2F%2Ftowardsdatascience.com%2F2326a3487cd5
    
    •	https://towardsdatascience.com/faq-build-a-handwritten-text-recognition-system-using-tensorflow-27648fb18519 
    


