import cv2
import os
import mediapipe as mp

# Inicializar MediaPipe Hands (para detectar manos)
mp_hands = mp.solutions.hands

# Configuración global
DATASET_DIR = 'LSM_dataset'
LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
NUM_IMAGES_PER_LETTER = 200
BOX_MARGIN = 30

def initialize_dataset_directory():
    """
    Crea el directorio base para almacenar el dataset si no existe.
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

def create_letter_directory(letter):
    """
    Crea un directorio para una letra específica si no existe.
    """
    letter_dir = os.path.join(DATASET_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)
    return letter_dir

def capture_images_for_letter(letter, num_images=NUM_IMAGES_PER_LETTER):
    """
    Captura imágenes de manos para una letra específica utilizando la cámara web.

    Args:
        letter (str): La letra para la cual se capturarán imágenes.
        num_images (int): Número de imágenes a capturar.
    """
    letter_dir = create_letter_directory(letter)

    # Inicializar la cámara web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara web.")
        return

    # Inicializar MediaPipe Hands
    hands = mp_hands.Hands()

    print(f"Presiona cualquier tecla para comenzar a capturar imágenes para la letra: {letter}")

    # Esperar a que el usuario presione una tecla
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el cuadro.")
            break

        frame = cv2.flip(frame, 1)  # Voltear el cuadro para evitar el efecto espejo
        cv2.putText(frame, "Presiona cualquier tecla para comenzar", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detección de Manos', frame)

        if cv2.waitKey(1) & 0xFF != 255:
            break

    print(f"Capturando imágenes para la letra: {letter}")
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el cuadro.")
            break

        frame = cv2.flip(frame, 1)  # Voltear el cuadro
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB

        # Detectar manos
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Expandir el cuadro delimitador
                x_min = max(x_min - BOX_MARGIN, 0)
                x_max = min(x_max + BOX_MARGIN, w)
                y_min = max(y_min - BOX_MARGIN, 0)
                y_max = min(y_max + BOX_MARGIN, h)

                # Dibujar el cuadro delimitador
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Recortar y guardar la región de la mano
                hand_region = frame[y_min:y_max, x_min:x_max]
                img_path = os.path.join(letter_dir, f'{letter}_{count}.jpg')
                cv2.imwrite(img_path, hand_region)
                count += 1

        # Mostrar la etiqueta de la letra
        cv2.putText(frame, f"Letra: {letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detección de Manos', frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Función principal para capturar imágenes de manos para cada letra.
    """
    initialize_dataset_directory()

    for letter in LETTERS:
        capture_images_for_letter(letter)

    print("¡Creación del dataset completa!")

if __name__ == "__main__":
    main()