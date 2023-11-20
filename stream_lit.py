import streamlit as st
from PIL import Image
import numpy
import cv2
from inference import ObjectDetectionClassification



st.subheader('Object Detection/Classification Module : Plant Disease')


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img 


def main():
    try:
        st.title("File Upload")
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            st.write(file_details)
            img = load_image(uploaded_file)
            image_open_cv = numpy.array(img)
            # Convert from RGB to BGR (OpenCV uses BGR format)
            cv2_img = cv2.cvtColor(image_open_cv, cv2.COLOR_BGR2RGB)
            cv2_img = cv2.resize(cv2_img,(640,640))
            st.image(img,width=256)
            if st.button('Predict'):
                obj = ObjectDetectionClassification(cv2_img)
                im = obj.main()
                st.image(im,width=500)
    except Exception as e:
        print(e)

if __name__=="__main__":
    main()