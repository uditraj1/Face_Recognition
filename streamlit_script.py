import base64
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
import cv2
import pandas as pd
from deepface import DeepFace
import numpy as np
import re
import os
import pdb
# from tempfile import NamedTemporaryFile

# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input

def findingFace(img):
    #uploaded_file = st.file_uploader("Choose File", type=["jpg", "png"])
    if img is not None:
        #image1 = img
        #st.image(image1)
        image1 = asarray(img)[:, :, ::-1]
        findFace = DeepFace.find(
        img_path=image1,
        db_path=os.getcwd() + "/new_dataset/",
        model_name="Facenet",
        enforce_detection=False,
        )
        nameList = []
        distanceList = []
        celebName = ""
        index = 0
        # no loop needed
        for findface in findFace: 
            findface.to_csv("Face"+str(index)+".csv")
            index = index + 1
        results = pd.read_csv('Face0.csv')
        for i in range(0,len(results)):
            #celebName = str(findFace[0].iloc[[i]].str[23:-3])
            celebName = str(findFace[0]["identity"].iloc[[i]].str[29:-3])
            matchingDistance = str((findFace[0]["Facenet_cosine"].iloc[[i]]))
            matchingDistance = matchingDistance[2:-19]
            celebNameTrunc = re.sub("[^A-Z]", " ", celebName, 0, re.IGNORECASE)
            matchingDistanceTrunc = re.sub("[A-Z:_]", " ", matchingDistance, 0, re.IGNORECASE)
                    #matchingDistanceTrunc = matchingDistanceTrunc.replace("Name: Facenet_cosine,", "")
                    #matchingDistanceTrunc = matchingDistanceTrunc.replace("dtype:float64", "")
                    #celebNameTrunc = celebNameTrunc.replace("D DeepFace new dataset", "")
                    #celebNameTrunc = celebNameTrunc.replace("jpg","")
            celebNameTrunc = celebNameTrunc.replace("Name", "")
            celebNameTrunc = celebNameTrunc.replace("identity", "")
            celebNameTrunc = celebNameTrunc.replace("dtype", "")
            celebNameTrunc = celebNameTrunc.replace("object", "")
            nameList.append(celebNameTrunc)
            distanceList.append(matchingDistanceTrunc)
        # add to dataframe
        res = "\n".join("{} {}".format(x, y) for x, y in zip(nameList, distanceList))
        #st.write("Closest Matches in Order:")
        #st.write(res)
        print("The Closest Matches In Order:")
        print(res)
        return res

colu1, colu2, colu3 = st.columns(3)

# with column1:
# st.write('')
# with column2:
st.title("Face Recognition App")
# with column3:
# st.write('')

choice = st.selectbox(
    "Select Among The Following:",
    ["Face Detection", "Face Verification", "Get Embeddings", "Find Face", "Face Analysis"],
)


def main():
    fig = plt.figure()
    if choice == "Face Detection":
        # load the image
        uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png"])
        if st.button("Detect Face"):
            if uploaded_file is not None:
                data = asarray(Image.open(uploaded_file))
                # plot the image
                plt.axis("off")
                plt.imshow(data)
                # get the context for drawing boxes
                ax = plt.gca()
                # plot each box
                # load image from file
                # create the detector, using default weights
                detector = MTCNN()
                # detect faces in the image
                faces = detector.detect_faces(data)
                for face in faces:
                    # get coordinates
                    x, y, width, height = face["box"]
                    # create the shape
                    rect = Rectangle((x, y), width, height, fill=False, color="maroon")
                    # draw the box
                    ax.add_patch(rect)
                    # draw the dots
                    for _, value in face["keypoints"].items():
                        # create and draw dot
                        dot = Circle(value, radius=2, color="maroon")
                        ax.add_patch(dot)
                # show the plot
                st.pyplot(fig)

    elif choice == "Face Verification":
        column1, column2 = st.columns(2)

        with column1:
            image3 = st.file_uploader("Choose First File", type=["jpg", "png"])

        with column2:
            image4 = st.file_uploader("Choose Second File", type=["jpg", "png"])
        # define filenames
        if (image3 is not None) & (image4 is not None):
            col1, col2 = st.columns(2)
            image1 = Image.open(image3)
            image2 = Image.open(image4)
            with col1:
                st.image(image1)
            with col2:
                st.image(image2)

            # filenames = [image1,image2]
            image1 = asarray(image1)[:, :, ::-1]
            image2 = asarray(image2)[:, :, ::-1]

            # im_cv = cv2.imread(image1)
            # cv2.imwrite(image1, im_cv)

            # im_cv1 = cv2.imread(image2)
            # cv2.imwrite(image2, im_cv1)

            # prepare the face for the model, e.g. center pixels
            # samples = preprocess_input(samples, version=2)

            if st.button("Verify Faces"):
                verification = DeepFace.verify(
                    img1_path=image1,
                    img2_path=image2,
                    model_name="Facenet",
                    detector_backend="opencv",
                    distance_metric="cosine",
                    align=True,
                    enforce_detection=True,
                )

                verification["verified"] = str(verification["verified"])

                print(verification["verified"])

                # create a vggface model

                # model = VGGFace(model= "resnet50" , include_top=False, input_shape=(224, 224, 3),
                # pooling= "avg" )

                # perform prediction
                # embeddings = model.predict(samples)

                # score = cosine(embeddings[0], embeddings[1])
                if verification["verified"] == "True":
                    st.success(" >FACE IS A MATCH")
                else:
                    st.error(" >FACE IS NOT A MATCH")

                return verification

    elif choice == "Face Analysis":
        # load the image
        uploaded_file = st.file_uploader("Choose File", type=["jpg", "png"])
        # with NamedTemporaryFile(dir=".", suffix='.jpg') as f:
        if uploaded_file is not None:
            image1 = Image.open(uploaded_file)
            st.image(image1)
            image1 = asarray(image1)[:, :, ::-1]
            # image1 = np.expand_dims(image1, axis=0)
            # image1 = np.reshape(image1, (48, 48, 3))
            # image1 = tf.reshape(image1, [675, 224])
            if st.button("Analyse Face"):
                analysis = DeepFace.analyze(
                    img_path=image1,
                    actions=["age", "gender", "emotion", "race"],
                    detector_backend="opencv",
                    enforce_detection=True,
                    align=True,
                )
                st.write(analysis)
                return analysis

    elif choice == "Get Embeddings":
        uploaded_file = st.file_uploader("Choose File For Embeddings", type=["jpg", "png"])
        if st.button("Display Embeddings Used"):
            if uploaded_file is not None:
                image1 = Image.open(uploaded_file)
                image1 = asarray(image1)[:, :, ::-1]
                result = {}
                embedding_objs = DeepFace.represent(
                    img_path=image1,
                    model_name="Facenet",
                    enforce_detection=True,
                    detector_backend="opencv",
                    align=True,
                    normalization="base",
                )
                #img = cv2.cvtColor(cv2.imread("deepface/tests/dataset/img1.jpg"), cv2.COLOR_BGR2RGB)
                #detector = MTCNN()
                #faces=detector.detect_faces(image1)    
                #pass2
                #embeddings = []
                #for face in faces:
                    #x, y, w, h = face["box"]
                    #detected_face = image1[int(y):int(y+h), int(x):int(x+w)]
                    #embedding = DeepFace.represent(img_path = detected_face, model_name = 'Facenet', enforce_detection = False)
                    #embeddings.append(embedding)
                result["results"] = embedding_objs
                st.write(result)
                return result

    elif choice == "Find Face":
        thresholds = {
        "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
        }
        uploaded_file = st.file_uploader("Choose File", type=["jpg", "png"])
        if uploaded_file is not None:
            image1 = Image.open(uploaded_file)
            st.image(image1)
            image1 = asarray(image1)[:, :, ::-1]
            db_path = os.getcwd() + "/new_dataset/"
            if st.button("Search For Face"):
                findFace = DeepFace.find(
                    img_path=image1,
                    db_path=db_path,
                    model_name="Facenet",
                    enforce_detection=False,
                )
                nameList = []
                distanceList = []
                celebName = ""
                celebName = os.path.split(str(findFace[0]["identity"].iloc[0]))[-1].split('.')[0]
                matchingDistance = float(findFace[0]["Facenet_cosine"].iloc[0])
                if matchingDistance < 0.40:
                    celebName = celebName.replace("_", " ")
                    nameList.append(celebName)
                    distanceList.append(matchingDistance)
                    # add to dataframe
                    res = "\n".join("{} {}".format(x, y) for x, y in zip(nameList, distanceList))
                    st.write("Closest Match In Database:")
                    st.write(res)
                    # return res
                else:
                    st.write("No Matching Faces For The Provided Threshold")

def extract_face(file):
    pixels = asarray(file)
    plt.axis("off")
    plt.imshow(pixels)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = asarray(image)
    return face_array


if __name__ == "__main__":
    main()
