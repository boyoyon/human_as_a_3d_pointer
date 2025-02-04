<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>3Dポインターとしての人</center></h1>
        <h2>なにものか？</h2>
        <p>
            Lightweight-Head-Pose-Estimationを使って3Dモデルを動かすだけのプログラムです。<br>
            <br>
            ・入力画像(左)から顔を検出し、顔の向きを抽出<br>
            ・出力画像(右)の向きを変更<br>
            <img src="images/input_output.gif"><br>
        </p>
        <h2>環境構築方法</h2>
        <p>
            <h3>[1] ベース環境をダウンロード～解凍～配置する</h3>
            　<a href="https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation">Lightweight-Head-Pose-Estimation</a><br>
            　Code → Download ZIP をクリックする。<br>
            <br>
              Lightweight-Head-Pose-Estimation-main.zip を解凍し、<br>
              Lightweight-Head-Pose-Estimation-main フォルダ内のファイル、フォルダを<br>
              src フォルダの下に配置する。<br>
            <br>
            <h3>[2] ライブラリをインストールする</h3>
            　・PyTorchをインストールする<br>
            　　手持ちのGPUの都合でv1.13.0でしか試しておりません。<br>
            　　<a href="https://pytorch.org/get-started/previous-versions/">https://pytorch.org/get-started/previous-versions/</a><br>
            　　v1.13.0 のpip install の手順を参照。<br>
            　・pip install opencv-python PyOpenGL glfw plyfile pillow<br>
            　・pip install numpy==1.26.1<br>
          </p>
        <h2>使い方</h2>
        <p>
            python src\RgbPly_rotated_by_face.py (RGB画像ファイル) (PLYファイル)<br>
            例) python src\RgbPly_rotated_by_face.py data\rgb1.png data\mesh1.png<br>
            <br>
            PLYファイルの作り方は<a href="https://github.com/boyoyon/RgbPly"> RgbPly </a>を参照。<br>
        </p>
    </body>
</html>
