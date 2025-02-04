import cv2, os, sys
import glfw
import numpy as np
import torch
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from plyfile import PlyData, PlyElement
from network.network import Network
from torchvision import transforms
from PIL import Image
from utils import load_snapshot
from utils.camera_normalize import drawAxis

#画像幅をALIGNピクセルの倍数にcropする
ALIGN = 4

# マウスドラッグ中かどうか
isDragging = False

# マウスのクリック位置
oldPos = [0, 0]
newPos = [0, 0]

# 操作の種類
MODE_NONE = 0x00
MODE_TRANSLATE = 0x01
MODE_ROTATE = 0x02
MODE_SCALE = 0x04

# マウス移動量と回転、平行移動の倍率
ROTATE_SCALE = 10.0
TRANSLATE_SCALE = 500.0

# 座標変換のための変数
Mode = MODE_NONE
RotMat = TransMat = ScaleMat = None
Scale = 1.0

# スキャンコード定義
SCANCODE_LEFT  = 331
SCANCODE_RIGHT = 333
SCANCODE_UP    = 328
SCANCODE_DOWN  = 336

# キーコード定義
KEY_I = 73 # toggle inertia mode 
KEY_S = 83 # screen shot
KEY_R = 82 # roll
KEY_Z = 90 # depth
KEY_P = 80 # print AZIMUTH, ELEVATION

MODS_SHIFT = 1

KEY_STATE_NONE = 0
KEY_STATE_PRESS_R = 4

keyState = KEY_STATE_NONE
PrevKeyState = KEY_STATE_NONE

# 方位角、仰角
AZIMUTH = 0.0
ELEVATION = 0.0
ROLL = 0.0
dAZIMUTH = 0.0
dELEVATION = 0.0


# FaceMeshをドロネー分解した三角形の頂点インデックスリスト
DEF_TRIANGLES = 'def_triangles2.txt'
NUM_DIVS = 20

# モデル位置
ModelPos = [0.0, 0.0]

# テクスチャー画像
textureImage = None

positions = []
texcoords = []
faces = []

WIN_WIDTH = 600  # ウィンドウの幅 / Window width
WIN_HEIGHT = 800  # ウィンドウの高さ / Window height
WIN_TITLE = "ImgPly to 3D"  # ウィンドウのタイトル / Window title

textureId = 0

idxModel = None

frameNo = 1

fInertia = False

def scale_bbox(bbox, scale):
    w = max(bbox[2], bbox[3]) * scale
    x= max(bbox[0] + bbox[2]/2 - w/2,0)
    y= max(bbox[1] + bbox[3]/2 - w/2,0)
    return np.asarray([x,y,w,w],np.int64)

def save_ply(path_ply, flag = 0):

    with open(path_ply, mode='w') as f:

        line = 'ply\n'
        f.write(line)

        line = 'format ascii 1.0\n'
        f.write(line)

        line = 'element vertex %d\n' % len(positions)
        f.write(line)

        line = 'property float x\n'
        f.write(line)

        line = 'property float y\n'
        f.write(line)

        line = 'property float z\n'
        f.write(line)

        line = 'property float s\n'
        f.write(line)

        line = 'property float t\n'
        f.write(line)

        line = 'element face %d\n' % len(faces)
        f.write(line)

        #line = 'property list uchar int vertex_index\n'
        line = 'property list uchar int vertex_indices\n'
        f.write(line)

        line = 'end_header\n'
        f.write(line)

        for i in range(len(positions)):
            line = '%f %f %f %f %f\n' % (
                    positions[i][0], positions[i][1], positions[i][2],
                    texcoords[i][0], texcoords[i][1])
            f.write(line)

        for i in range(len(faces)):
            idx0 = faces[i][0]
            idx1 = faces[i][1]
            idx2 = faces[i][2]
            
            if flag == 0:
                line = '3 %d %d %d\n' % (idx0, idx1, idx2)
            else:
                line = '3 %d %d %d\n' % (idx0, idx2, idx1)
            
            f.write(line)

# OpenGLの初期化関数
def initializeGL():
    global textureId, idxModel, lightPos

    # 背景色の設定 (黒)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # 深度テストの有効化
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_CULL_FACE)

    # テクスチャの有効化
    glEnable(GL_TEXTURE_2D)

    # テクスチャの設定

    image = textureImage

    texHeight, texWidth, _ = image.shape

    # テクスチャの生成と有効化
    textureId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureId)

    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8, texWidth, texHeight, GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

    # テクスチャ境界の折り返し設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # テクスチャの無効化
    glBindTexture(GL_TEXTURE_2D, 0)

    idxModel = glGenLists(1)

    glNewList(idxModel, GL_COMPILE)

    glBegin(GL_TRIANGLES)

    for i in range(len(faces)):

        idx0 = faces[i][0]
        idx1 = faces[i][1]
        idx2 = faces[i][2]

        glTexCoord2fv(texcoords[idx0])
        glVertex3fv(positions[idx0])

        glTexCoord2fv(texcoords[idx1])
        glVertex3fv(positions[idx1])

        glTexCoord2fv(texcoords[idx2])
        glVertex3fv(positions[idx2])

    glEnd()

    glEndList()

# OpenGLの描画関数
def paintGL():

    if WIN_HEIGHT > 0:

        # 背景色と深度値のクリア
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
        # 投影変換行列
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(10.0, WIN_WIDTH / WIN_HEIGHT, 1.0, 100.0)
    
        # モデルビュー行列
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        gluLookAt(0.0, 0.0, -5.0,   # 視点の位置
            0.0, 0.0, 0.0,   # 見ている先
            0.0, -1.0, 0.0)  # 視界の上方向
    
        # 平面の描画
        glBindTexture(GL_TEXTURE_2D, textureId)  # テクスチャの有効化
    
        glPushMatrix()
        glScalef(Scale, Scale, Scale)
        glTranslatef(ModelPos[0], ModelPos[1], 0.0)
        glRotatef(ELEVATION, 1.0, 0.0, 0.0)
        glRotatef(AZIMUTH, 0.0, 1.0, 0.0)
        glRotatef(ROLL, 0.0, 0.0, 1.0)
        glCallList(idxModel)
        glPopMatrix()
    
        glBindTexture(GL_TEXTURE_2D, 0)  # テクスチャの無効化


# ウィンドウサイズ変更のコールバック関数
def resizeGL(window, width, height):
    global WIN_WIDTH, WIN_HEIGHT

    # ユーザ管理のウィンドウサイズを変更
    WIN_WIDTH = width
    WIN_HEIGHT = height

    # GLFW管理のウィンドウサイズを変更
    glfw.set_window_size(window, WIN_WIDTH, WIN_HEIGHT)

    # 実際のウィンドウサイズ (ピクセル数) を取得
    renderBufferWidth, renderBufferHeight = glfw.get_framebuffer_size(window)

    # ビューポート変換の更新
    glViewport(0, 0, renderBufferWidth, renderBufferHeight)

# アニメーションのためのアップデート
def animate():
    global AZIMUTH, ELEVATION

    if fInertia and not isDragging:
        AZIMUTH -= dAZIMUTH
        ELEVATION += dELEVATION

def save_screen(window):
    global frameNo

    width = WIN_WIDTH
    height = WIN_HEIGHT

    if width % 4 != 0:
        width = width // 4 * 4
        resizeGL(window, width, height)

    glReadBuffer(GL_FRONT)
    screen_shot = np.zeros((height, width, 3), np.uint8)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_shot.data)
    screen_shot = cv2.flip(screen_shot, 0) 
    screen_shot = cv2.cvtColor(screen_shot, cv2.COLOR_RGB2BGR)
    filename = 'screenshot_%04d.png' % frameNo
    cv2.imwrite(filename, screen_shot)
    print('saved %s' % filename)
    
    depth_shot = np.zeros((height, width), np.uint16)
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, depth_shot.data)
    #depth_shot = np.zeros((height, width), np.float32)
    #glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth_shot.data)
    depth_bias = np.min(depth_shot)
    depth_scale = np.max(depth_shot) - np.min(depth_shot)

    frameNo += 1

# キーボードの押し離しを扱うコールバック関数
def keyboardEvent(window, key, scancode, action, mods):
    global AZIMUTH, ELEVATION, keyState, dAZIMUTH, dELEVATION, fInertia, Scale

    # 矢印キー操作

    if scancode == SCANCODE_LEFT:
        dAZIMUTH = -0.1
        AZIMUTH -= dAZIMUTH * 10

    if scancode == SCANCODE_RIGHT:
        dAZIMUTH = 0.1
        AZIMUTH -= dAZIMUTH * 10

    if scancode == SCANCODE_DOWN:
        dELEVATION = 0.1
        ELEVATION += dELEVATION * 10

    if scancode == SCANCODE_UP:
        dELEVATION = -0.1
        ELEVATION += dELEVATION * 10

    if key == KEY_Z:
        if mods == MODS_SHIFT:
            Scale += 0.1
        else:
            Scale -= 0.1

    if action == 1: #press key
        if key == KEY_S:
            save_screen(window)

        if key == KEY_I:
            fInertia = not fInertia

        if key == KEY_P:
            print('[%f, %f],' % (AZIMUTH, ELEVATION))

    if key == KEY_R:
        if action == glfw.PRESS:
            keyState = KEY_STATE_PRESS_R
        elif action == 0:
            keyState = KEY_STATE_NONE

# マウスのクリックを処理するコールバック関数

def mouseEvent(window, button, action, mods):
    global isDragging, newPos, oldPos, Mode, fInertia

    # クリックしたボタンで処理を切り替える
    if button == glfw.MOUSE_BUTTON_LEFT:
        Mode = MODE_ROTATE
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        Mode = MODE_SCALE
        if action == 1:
            fInertia = not fInertia

    elif button == glfw.MOUSE_BUTTON_RIGHT:
        Mode = MODE_TRANSLATE

    # クリックされた位置を取得
    px, py = glfw.get_cursor_pos(window)

    # マウスドラッグの状態を更新
    if action == glfw.PRESS:
        if not isDragging:
            isDragging = True
            oldPos = [px, py]
            newPos = [px, py]
    else:
        isDragging = False
        oldPos = [0, 0]
        newPos = [0, 0]

# マウスの動きを処理するコールバック関数
def motionEvent(window, xpos, ypos):
    global isDragging, newPos, oldPos, AZIMUTH, ELEVATION, ModelPos, dAZIMUTH, dELEVATION

    if isDragging:
        # マウスの現在位置を更新
        newPos = [xpos, ypos]

        # マウスがあまり動いていない時は処理をしない
        dx = newPos[0] - oldPos[0]
        dy = newPos[1] - oldPos[1]
        #length = dx * dx + dy * dy
        #if length < 2.0 * 2.0:
        #    return
        #else:
        if Mode == MODE_ROTATE:
            dAZIMUTH = (xpos - oldPos[0]) / ROTATE_SCALE
            dELEVATION = (ypos - oldPos[1]) / ROTATE_SCALE
            AZIMUTH -= dAZIMUTH
            ELEVATION += dELEVATION
        elif Mode == MODE_TRANSLATE:
            ModelPos[0] += (xpos - oldPos[0]) / TRANSLATE_SCALE
            ModelPos[1] += (ypos - oldPos[1]) / TRANSLATE_SCALE

        oldPos = [xpos, ypos]

# マウスホイールを処理するコールバック関数
def wheelEvent(window, xoffset, yoffset):
    global Scale, ROLL

    if keyState == KEY_STATE_NONE:
        Scale += yoffset / 10.0

    elif keyState == KEY_STATE_PRESS_R:
        ROLL += yoffset

# 画像の横幅がALIGNピクセルの倍数になるようにクロップする
def prescale(image):
    height, width = image.shape[:2]

    if width % ALIGN != 0:
        WIDTH = width // ALIGN * ALIGN
        startX = (width - WIDTH) // 2
        endX = startX + WIDTH

        dst = np.empty((height, WIDTH, 3), np.uint8)
        dst = image[:, startX:endX]
        return dst

    else:
        return image

def main():

    global textureImage, positions, texcoords, faces, WIN_WIDTH, WIN_HEIGHT
    global AZIMUTH, ELEVATION, ROLL

    argv = sys.argv
    argc = len(argv)

    if argc < 3:
        print('%s displays texture mapped PLY' % argv[0])
        print('%s <image> <ply> [,zScale>]' % argv[0])
        quit()

    img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
    img = prescale(img)
    WIN_HEIGHT, WIN_WIDTH = img.shape[:2]

    if WIN_HEIGHT < 256 and WIN_WIDTH < 256:
        WIN_HEIGHT *= 2
        WIN_WIDTH *= 2

    textureImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    textureImage = cv2.flip(textureImage, 0)

    plydata = PlyData.read(argv[2])
   
    zScale = 1.0
    
    if argc > 3:
        zScale = float(argv[3])

    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z'] * zScale
    s = plydata.elements[0].data['s']
    t = plydata.elements[0].data['t']
    
    meanX = np.mean(x)
    meanY = np.mean(y)
    meanZ = np.mean(z)
    
    for i in range(x.shape[0]):
        positions.append((x[i]-meanX, y[i]-meanY, z[i]-meanZ))
        texcoords.append((s[i], t[i]))

    for i in range(plydata['face'].data['vertex_indices'].shape[0]):
        faces.append(plydata['face'].data['vertex_indices'][i].tolist())

    # OpenGLを初期化する
    if glfw.init() == glfw.FALSE:
        raise Exception("Failed to initialize OpenGL")

    # Windowの作成
    window = glfw.create_window(WIN_WIDTH, WIN_HEIGHT, WIN_TITLE, None, None)
    if window is None:
        glfw.terminate()
        raise Exception("Failed to create Window")

    # OpenGLの描画対象にWindowを追加
    glfw.make_context_current(window)

    # ウィンドウのリサイズを扱う関数の登録
    glfw.set_window_size_callback(window, resizeGL)

    # キーボードのイベントを処理する関数を登録
    glfw.set_key_callback(window, keyboardEvent)

    # マウスのイベントを処理する関数を登録
    glfw.set_mouse_button_callback(window, mouseEvent)

    # マウスの動きを処理する関数を登録
    glfw.set_cursor_pos_callback(window, motionEvent)

    # マウスホイールを処理する関数を登録
    glfw.set_scroll_callback(window, wheelEvent)
    
    # ユーザ指定の初期化
    initializeGL()

    cap = cv2.VideoCapture(0)

    face_detector_model = os.path.join(os.path.dirname(__file__), 'lbpcascade_frontalface_improved.xml')
    face_cascade = cv2.CascadeClassifier(face_detector_model)

    pose_estimator = Network(bin_train=False)
     
    pose_estimator_model = os.path.join(os.path.dirname(__file__), './models/model-b66.pkl')
    load_snapshot(pose_estimator, pose_estimator_model)
    pose_estimator = pose_estimator.eval()

    transform_test = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    count = 0
    last_faces = None

    print('Hit ESC-key or q-key repeatedly to terminate this program')

    while True:
        
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if count % 5 == 0:
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, 1.2)
            if len(faces)==0 and (last_faces is not None):
                faces=last_faces
            last_faces = faces

        face_images = []
        face_tensors = []
        for i, bbox in enumerate(faces):
            x,y, w,h = scale_bbox(bbox,1.5)
            frame = cv2.rectangle(frame,(x,y), (x+w, y+h),color=(0,0,255),thickness=2)
            face_img = frame[y:y+h,x:x+w]
            face_images.append(face_img)
            pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(face_img,(224,224)), cv2.COLOR_BGR2RGB))
            face_tensors.append(transform_test(pil_img)[None])

        if len(face_tensors)>0:
            with torch.no_grad():
                face_tensors = torch.cat(face_tensors,dim=0)
                roll, yaw, pitch = pose_estimator(face_tensors)
                
                ROLL = roll
                AZIMUTH = yaw
                ELEVATION = -pitch

                for img, r, y, p in zip(face_images, roll, yaw, pitch):
                    headpose = [r, y, p]
                    drawAxis(img, headpose,size=50)

        cv2.imshow("Result", frame)

        # 描画
        paintGL()

        # アニメーション
        animate()

        # 描画用バッファの切り替え
        glfw.swap_buffers(window)
        glfw.poll_events()

        key = cv2.waitKey(10)
        if key==27 or key == ord("q"):
            break
        count+=1

    # 後処理
    glfw.destroy_window(window)
    glfw.terminate()
 
    cap.release
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
