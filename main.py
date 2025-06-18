import cv2
import numpy as np
import random
import os
import time
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def create_training_dirs(emojis, base_dir="training_data"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for emoji_name in emojis:
        path = os.path.join(base_dir, emoji_name)
        if not os.path.exists(path):
            os.makedirs(path)
    print(f"Diretórios de treino criados em: {base_dir}")

def extract_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)
            return np.array(landmarks)
        return None

def load_and_process_training_data(emoji_map, base_dir="training_data"):
    X = [] # Features (landmarks)
    y = [] # Labels (emoji names)
    for emoji_char, emoji_name in emoji_map.items():
        path = os.path.join(base_dir, emoji_name)
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(path, filename)
                    image = cv2.imread(img_path)
                    if image is not None:
                        landmarks = extract_landmarks(image)
                        if landmarks is not None:
                            X.append(landmarks)
                            y.append(emoji_name)
    return np.array(X), np.array(y)

def train_model(X, y):
    if len(X) == 0:
        print("Nenhum dado de treino encontrado. Não é possível treinar o modelo.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel=\'linear\', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Acurácia do modelo: {accuracy_score(y_test, y_pred)}")
    return model

def main():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam. Certifique-se de que ela está conectada e não está sendo usada por outro aplicativo.")
        return

    emojis = ["😀", "😂", "😍", "🤔", "😡", "👍", "👎", "👏", "🙏", "🤩"]
    emoji_names = ["sorrindo", "chorando_de_rir", "apaixonado", "pensativo", "bravo", "joinha", "nao_joinha", "aplauso", "rezando", "estrela"]
    emoji_map = dict(zip(emojis, emoji_names))
    reverse_emoji_map = dict(zip(emoji_names, emojis))

    create_training_dirs(emoji_names)

    mode = "menu" # "menu", "training", "game"
    current_emoji_to_train = None
    capture_count = 0
    last_capture_time = 0
    capture_interval = 0.5 # seconds between captures

    model = None
    model_filename = "emoji_recognition_model.pkl"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0)  # Green color

    # Game variables
    game_emojis = []
    current_game_round = 0
    score = 0
    round_start_time = 0
    round_duration = 30 # seconds
    game_state_message = ""

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da webcam.")
                break

            frame = cv2.flip(frame, 1) # Flip horizontally for selfie-view
            display_frame = frame.copy()
            display_text = ""

            if mode == "menu":
                display_text = "Pressione \'T\' para Treinar, \'J\' para Jogar, \'L\' para Carregar Modelo"
                text_size = cv2.getTextSize(display_text, font, font_scale, font_thickness)[0]
                text_x = (display_frame.shape[1] - text_size[0]) // 2
                text_y = (display_frame.shape[0] + text_size[1]) // 2
                cv2.putText(display_frame, display_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            elif mode == "training":
                if current_emoji_to_train is None:
                    display_text = "Selecione um emoji para treinar (1-10):\n"
                    for i, emoji in enumerate(emojis):
                        display_text += f"{i+1}. {emoji} ({emoji_map[emoji]})\n"
                    # This will be drawn line by line later
                else:
                    display_text = f"Treinando: {current_emoji_to_train} ({emoji_map[current_emoji_to_train]})\nPressione \'C\' para Capturar, \'M\' para Menu, \'S\' para Salvar Modelo\nCapturas: {capture_count}"
                    text_size = cv2.getTextSize(display_text.split(\'\n\')[0], font, font_scale, font_thickness)[0]
                    text_x = (display_frame.shape[1] - text_size[0]) // 2
                    text_y = (display_frame.shape[0] + text_size[1]) // 2
                    cv2.putText(display_frame, display_text.split(\'\n\')[0], (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(display_frame, display_text.split(\'\n\')[1], (text_x, text_y + 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(display_frame, display_text.split(\'\n\')[2], (text_x, text_y + 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            elif mode == "game":
                if model is None:
                    display_text = "Modelo não treinado/carregado. Treine ou carregue um modelo primeiro!"
                    text_size = cv2.getTextSize(display_text, font, font_scale, font_thickness)[0]
                    text_x = (display_frame.shape[1] - text_size[0]) // 2
                    text_y = (display_frame.shape[0] + text_size[1]) // 2
                    cv2.putText(display_frame, display_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                else:
                    if current_game_round < 10:
                        elapsed_time = time.time() - round_start_time
                        remaining_time = max(0, round_duration - int(elapsed_time))

                        target_emoji_char = game_emojis[current_game_round]
                        target_emoji_name = emoji_map[target_emoji_char]

                        # Real-time prediction
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_frame)
                        predicted_emoji_char = "?"
                        if results.multi_face_landmarks:
                            landmarks = []
                            for landmark in results.multi_face_landmarks[0].landmark:
                                landmarks.append(landmark.x)
                                landmarks.append(landmark.y)
                                landmarks.append(landmark.z)
                            predicted_emoji_name = model.predict([np.array(landmarks)])[0]
                            predicted_emoji_char = reverse_emoji_map.get(predicted_emoji_name, "?")

                            # Draw landmarks
                            for face_landmarks in results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=display_frame,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                                mp_drawing.draw_landmarks(
                                    image=display_frame,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                            if predicted_emoji_name == target_emoji_name:
                                score += 1
                                game_state_message = "ACERTOU!"
                                current_game_round += 1
                                round_start_time = time.time() # Reset timer for next round
                                if current_game_round < 10:
                                    print(f"Rodada {current_game_round} de 10. Pontuação: {score}")
                                else:
                                    print(f"Fim de jogo! Pontuação final: {score}/10")
                                    mode = "menu"

                        if remaining_time <= 0:
                            game_state_message = "TEMPO ESGOTADO!"
                            current_game_round += 1
                            round_start_time = time.time() # Reset timer for next round
                            if current_game_round < 10:
                                print(f"Rodada {current_game_round} de 10. Pontuação: {score}")
                            else:
                                print(f"Fim de jogo! Pontuação final: {score}/10")
                                mode = "menu"

                        display_text = f"Rodada: {current_game_round + 1}/10 | Tempo: {remaining_time}s | Pontos: {score}\nImite: {target_emoji_char} | Você: {predicted_emoji_char}\n{game_state_message}"

                    else:
                        display_text = f"Fim de jogo! Pontuação final: {score}/10\nPressione \'M\' para Menu"

                    # Draw multi-line text
                    y_offset = 50
                    for line in display_text.split(\'\n\'):
                        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                        text_x = (display_frame.shape[1] - text_size[0]) // 2
                        cv2.putText(display_frame, line, (text_x, y_offset), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                        y_offset += 30

            if mode == "training" and current_emoji_to_train is None:
                y_offset = 50
                for line in display_text.split(\'\n\'):
                    text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                    text_x = (display_frame.shape[1] - text_size[0]) // 2
                    cv2.putText(display_frame, line, (text_x, y_offset), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    y_offset += 30

            cv2.imshow("Webcam App", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if mode == "menu":
                if key == ord("t"):
                    mode = "training"
                    current_emoji_to_train = None
                    capture_count = 0
                elif key == ord("j"):
                    if model is not None:
                        mode = "game"
                        game_emojis = random.sample(emojis, 10) # 10 random emojis for the game
                        current_game_round = 0
                        score = 0
                        round_start_time = time.time()
                        game_state_message = ""
                        print("Iniciando jogo!")
                    else:
                        print("Por favor, treine ou carregue um modelo antes de iniciar o jogo.")
                elif key == ord("l"):
                    if os.path.exists(model_filename):
                        with open(model_filename, \'rb\') as f:
                            model = pickle.load(f)
                        print("Modelo carregado com sucesso!")
                    else:
                        print("Nenhum modelo encontrado para carregar.")
            elif mode == "training":
                if current_emoji_to_train is None:
                    if ord(\'1\') <= key <= ord(\'9\'):
                        idx = int(chr(key)) - 1
                        if 0 <= idx < len(emojis):
                            current_emoji_to_train = emojis[idx]
                            capture_count = 0
                    elif key == ord(\'0\'): # For emoji 10
                        idx = 9
                        if 0 <= idx < len(emojis):
                            current_emoji_to_train = emojis[idx]
                            capture_count = 0
                else:
                    if key == ord("c"):
                        current_time = time.time()
                        if current_time - last_capture_time > capture_interval:
                            emoji_dir = os.path.join("training_data", emoji_map[current_emoji_to_train])
                            filename = os.path.join(emoji_dir, f"image_{int(time.time())}_{capture_count}.png")
                            cv2.imwrite(filename, frame)
                            capture_count += 1
                            last_capture_time = current_time
                            print(f"Capturada imagem para {current_emoji_to_train}: {filename}")
                    elif key == ord("s"):
                        X, y = load_and_process_training_data(emoji_map)
                        model = train_model(X, y)
                        if model:
                            with open(model_filename, \'wb\') as f:
                                pickle.dump(model, f)
                            print("Modelo salvo com sucesso!")
                    elif key == ord("m"):
                        mode = "menu"
                        current_emoji_to_train = None
            elif mode == "game":
                if key == ord("m"):
                    mode = "menu"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == \'__main__\':
    main()


