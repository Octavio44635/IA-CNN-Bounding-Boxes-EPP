
import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

# --- VARIABLES GLOBALES ---
# Flag para saber si ya está hablando
is_speaking = False
# Marca de tiempo de la última vez que habló
last_speech_time = 0
# Tiempo de espera entre mensajes (en segundos)
SPEECH_COOLDOWN = 5 

def speak_function(message):
    """
    Función que se ejecuta en un hilo separado.
    Inicializa el motor de voz LOCALMENTE cada vez para evitar
    bloqueos COM en Windows.
    """
    global is_speaking
    try:
        # Inicialización local (El truco anti-crashes de Windows)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) 
        engine.setProperty('volume', 1.0)
        
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Error de audio: {e}")
    finally:
        # Liberamos el candado
        is_speaking = False

def trigger_voice_warning(message):
    """
    Lanza el hilo de voz solo si:
    1. No está hablando actualmente.
    2. Ha pasado suficiente tiempo desde la última vez (Cooldown).
    """
    global is_speaking, last_speech_time
    
    current_time = time.time()
    
    # Verificamos si pasó el tiempo de espera (ej: 5 segundos)
    if not is_speaking and (current_time - last_speech_time > SPEECH_COOLDOWN):
        is_speaking = True
        last_speech_time = current_time
        
        # Lanzamos el hilo
        t = threading.Thread(target=speak_function, args=(message,))
        t.start()

def main():
    # --- 1. CARGA DEL MODELO ---
    # Ajusta esta ruta si es necesario. Si está en la misma carpeta, basta con el nombre.
    model_path = r"runs/segment/mi_modelo_epp_medium/weights/best.pt" 
    
    print(f"Cargando modelo desde: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        print("Verifica que la ruta del archivo .pt sea correcta.")
        return

    # --- 2. CONFIGURACIÓN DE CLASES ---
    print("Clases del modelo:", model.names)
    
    # IMPORTANTE: Verifica que el ID 0 sea realmente lo que buscas (ej. Casco)
    # Si necesitas Casco (0) Y Chaleco (1), usa: {0, 1}
    CLASES_OBLIGATORIAS = {0,1} 
    
    # --- 3. INICIO DE VIDEO ---
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Sistema iniciado. Presiona 'q' para salir.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # --- 4. INFERENCIA ---
        # conf=0.5: Filtra detecciones con baja confianza
        results = model(frame, verbose=False, conf=0.5)
        r = results[0]

        # Convertir clases detectadas a un conjunto de números enteros
        if r.boxes:
            clases_detectadas = set(r.boxes.cls.cpu().numpy().astype(int))
        else:
            clases_detectadas = set()

        # --- 5. LÓGICA DE SEGURIDAD ---
        # Verificamos si TODAS las clases obligatorias están presentes
        if CLASES_OBLIGATORIAS.issubset(clases_detectadas):
            # --- ESTADO: SEGURO ---
            status_text = "OBRERO SEGURO"
            status_color = (0, 255, 0) # Verde
            
            # Opcional: Que diga "seguro" también (puede ser molesto, comentar si se desea)
            trigger_voice_warning("OBRERO SEGURO.") 
        else:
            # --- ESTADO: PELIGRO ---
            status_text = "ALERTA: FALTA EPP"
            status_color = (0, 0, 255) # Rojo
            
            # Alerta de voz (respetando el cooldown de 5 seg)
            trigger_voice_warning("Alerta. Falta equipo de protección.")

        # --- 6. VISUALIZACIÓN ---
        annotated_frame = r.plot()

        # Fondo negro para el texto superior
        cv2.rectangle(annotated_frame, (0, 0), (640, 60), (0, 0, 0), -1)
        
        # Texto de estado
        cv2.putText(annotated_frame, status_text, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Debug: Mostrar qué está viendo realmente
        nombres = [model.names[i] for i in clases_detectadas]
        texto_debug = f"Viendo: {', '.join(nombres)}"
        cv2.putText(annotated_frame, texto_debug, (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Sistema de Monitoreo EPP", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()