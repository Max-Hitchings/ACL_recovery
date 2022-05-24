import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

IMG_PATH = ["img/vid_2_Moment.png","img/vid_Moment1.jpg", "img/vid_Moment2.jpg", "img/vid_Moment3.jpg", "img/vid_Moment4.jpg"]

# dim = (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25))
#
#
def show_img(img_src, scale: tuple = None, name: str = ""):
    if not scale:
        scale = (1, 1)

    img = cv2.imread(img_src)

    dim = (int(img.shape[1] * scale[0]), int(img.shape[0] * scale[1]))

    title = f"{name} {int(img.shape[1] * scale[0])} x {int(img.shape[0] * scale[1])}"

    cv2.imshow(title, cv2.resize(img, dim))
    cv2.waitKey(0)

    return 0


img_src = "img/vid_Moment1.jpg"
def do_stuff(img_src):
    with mp_pose.Pose(
        static_image_mode=True, model_complexity=2, enable_segmentation=True) as pose:

        # img = cv2.imread(img_src)
        img = cv2.imread(img_src)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = img.shape
        result = pose.process(img)

        for x in [25, 27, 29, 31]:


            center_coordinates = (int(result.pose_landmarks.landmark[x].x * image_width), int(result.pose_landmarks.landmark[x].y * image_height))
            # print(center_coordinates)
            img = cv2.circle(img, center_coordinates, 4, (255, 0, 0), 4)

        y2 = result.pose_landmarks.landmark[27].y *image_height
        x2 = result.pose_landmarks.landmark[27].x *image_width
        y1 = result.pose_landmarks.landmark[25].y  *image_height
        x1 = result.pose_landmarks.landmark[25].x  *image_width
        p = (y1 - y2)
        q = (x1 - x2)
        gradient = p / q
        # print("gradient 1", gradient)
        m1 = gradient

        yy1 = result.pose_landmarks.landmark[29].y  *image_height
        xx1 = result.pose_landmarks.landmark[29].x  *image_width
        yy2 = result.pose_landmarks.landmark[31].y  *image_height
        xx2 = result.pose_landmarks.landmark[31].x  *image_width
        gradient2 = (yy2-yy1) / (xx2-xx1)
        # print("gradient 2", gradient)
        m2 = gradient2

        angle = abs((m2 - m1) / (1 + m2 * m1))
        angle = math.degrees(math.atan(angle))
        angle = 180 - angle
        print( (m1 - m2), "/", (1 + m2 * m1))

        offset = -abs(angle-90)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(round(offset, 2)), (0,1800), font, 3, (22, 15, 117), 7, cv2.LINE_AA)
        cv2.putText(img, str(angle), (0,1900), font, 3, (0, 51, 0), 7, cv2.LINE_AA)


        # center_coordinates = (int(result.pose_landmarks.landmark[29].x ), int(result.pose_landmarks.landmark[29].y ))
        # img = cv2.circle(img, center_coordinates, 4, (255, 0, 0), 4)
        #
        # center_coordinates = (int(result.pose_landmarks.landmark[27].x *1.1), int(result.pose_landmarks.landmark[27].y *1.1))
        # img = cv2.circle(img, center_coordinates, 4, (255, 0, 0), 4)
        #
        # center_coordinates = (int(result.pose_landmarks.landmark[25].x ), int(result.pose_landmarks.landmark[25].y ))
        # img = cv2.circle(img, center_coordinates, 4, (255, 0, 0), 4)


        # mp_drawing.draw_landmarks(img, result.pose_landmarks.landmark[1].x * image_width, result.pose_landmarks.landmark[1].y * image_width)

        scale = 0.5
        dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("title", cv2.resize(img, dim))
        cv2.waitKey(0)


for x in IMG_PATH:
    print(x)
    do_stuff(x)




#
#
# def find_ancle_angle(img) -> float:
#     return 0.0
#
#
# if __name__ == "__main__":
#     show_img("./img/IMG_7857.png", (0.25, 0.25))
