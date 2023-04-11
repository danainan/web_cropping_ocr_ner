import cv2
import numpy as np
import streamlit as st
import keyboard
from PIL import Image
from tesserocr import PyTessBaseAPI
import streamlit.components.v1 as components
from transformers import AutoTokenizer , AutoModelForTokenClassification
from pythainlp.tokenize import word_tokenize
import torch

@st.cache_data()
def original_image(crop):
    load_image(crop)
    org = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    st.write('Original Image')
    st.image(org)
    
@st.cache_data()
def binary(crop):
    load_image(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    st.write('Binary Image')
    st.image(thresh)
    
@st.cache_data()
def dilation(crop):
    load_image(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.array([ [1, 0, 0],
                        [0, 1,0]],np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    st.write('Dilation Image')
    st.image(img_dilation)

@st.cache_data()
def erosion(crop):
    load_image(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.array([ [1, 0, 0],
                        [0, 1,0]],np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    st.write('Erosion Image')
    st.image(img_erosion)

@st.cache_data()
def opening(crop):
    load_image(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.array([ [1, 0, 0],
                        [0, 1,0]],np.uint8)
    img_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    st.image(img_opening)

@st.cache_data()
def closing(crop):
    load_image(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.array([ [1, 0, 0],
                        [0, 1,0]],np.uint8)
    img_closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    st.write('Closing Image')
    st.image(img_closing)

@st.cache_data()
def crop_image(crop):
    return crop

@st.cache_resource()
def load_image(image):
    return image


cap = cv2.VideoCapture(2)
frame = np.zeros((1280,720,3), dtype=np.uint8)


cap.set(3,1280)
cap.set(4,720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

x, y, w, h = 100, 100, 450, 250
w = st.slider('Width', 0, 1280, 450)
h = st.slider('Height', 0, 720, 250)
st.write('Press "q" to crop the image')
#stframe = st.empty()
stop = False

crop = frame[y:y+h, x:x+w]
reset = st.button('Reset')

#image = st.empty()

if crop is not None:
    stframe = st.empty()

    while stop == False :
        ret, frame = cap.read()
        if not ret:
            break
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        stframe.image(frame, channels="BGR")
        
        if cv2.waitKey(1) & keyboard.is_pressed('q'):
            stop = True
            break

        

    cap.release()
    cv2.destroyAllWindows()



    crop = frame[y:y+h, x:x+w]
    crop_image(crop)


    options = st.sidebar.selectbox('Select the operation', ('Original Image', 'Binary', 'Dilation', 'Erosion', 'Opening', 'Closing'))

    if options == 'Original Image':
        crop_image(crop)
        original_image(crop)
    elif options == 'Binary':
        crop_image(crop)
        binary(crop)
    elif options == 'Dilation':
        crop_image(crop)
        dilation(crop)
    elif options == 'Erosion':
        crop_image(crop)
        erosion(crop)
    elif options == 'Opening':
        crop_image(crop)
        opening(crop)
    elif options == 'Closing':
        crop_image(crop)
        closing(crop)
    
   
    with PyTessBaseAPI(path='C:/Users/User/anaconda3/share/tessdata_best-main',lang="tha+eng") as api:
        im_ocr = Image.fromarray(crop)
        api.SetImage(im_ocr)
        text = api.GetUTF8Text()
        text_array = []
        text_array.append(text.replace("\n", " "))
        st.write(text_array)

        name = 'thainer-corpus-v2-base-model'
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForTokenClassification.from_pretrained(name)

        sentence = text

        cut=word_tokenize(sentence.replace(" ", "<_>"))
        inputs=tokenizer(cut,is_split_into_words=True,return_tensors="pt")

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # forward pass
        outputs = model(ids, attention_mask=mask)
        logits = outputs[0]

        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

        def fix_span_error(words,ner):
            _ner = []
            _ner=ner
            _new_tag=[]
            for i,j in zip(words,_ner):
                #print(i,j)
                i=tokenizer.decode(i)
                if i.isspace() and j.startswith("B-"):
                    j="O"
                if i=='' or i=='<s>' or i=='</s>':
                    continue
                if i=="<_>":
                    i=" "
                _new_tag.append((i,j))
            return _new_tag

        ner_tag=fix_span_error(inputs['input_ids'][0],predicted_token_class)
        print(ner_tag)
        
        def merge_name(ner_tag):
            _new_tag=[]
            _phrase=''
            _tag=''
            for i,j in ner_tag:
                if j.startswith("B-"):
                    _phrase+=i
                    _tag=j
                elif j.startswith("I-"):
                    _phrase+=i
                else:
                    if _phrase!='':
                        _new_tag.append((_phrase,_tag))
                    _phrase=''
                    _tag=''
            return _new_tag

        ner_tag=merge_name(ner_tag)
        print('NAME NER ====>',ner_tag)

        person = []
        for i in ner_tag:
            if i[1].endswith("PERSON"):
                person.append(i[0])

        print('ผู้ส่ง :',person[0])
        print('ผู้รับ :',person[1])

        # st.write('ผู้ส่ง :',person[0])
        # st.write('ผู้รับ :',person[1])












    

