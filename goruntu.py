import  cv2
import imageio#resimler üzerinden kullanmak amaçlı

#cascade yüklemesi yapılır

face_cascade    =   cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade    =   cv2.CascadeClassifier('haarcascade_eye.xml')

#yüz ve göz için fonksiyon yazılır
def detect(frame):
    gray    =   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#renkler siyah beyaza çevrilir
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)#yüz araması yapılıyor gray siyah beyaza çevrilen resim, 1.3 küçültme oranı, 5 5 pencere tespit ederse orda yüz vardır
    for(x,  y,  w,  h) in faces:
        cv2.rectangle(frame,   (x,  y), (x+w,   y+h),   (255,   0,  0), 2)#frame resim alındı, kare çizildi ve ardından karenin rengi belirlendi, ve karenin çerçeve kalınlıgı çıkarıtıldı
        gray_face   =   gray[y:y+h, x:x+w]
        color_face = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)#yüz üzerinde arama yapılıyor
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face,  (ex,    ey),    (ex+ew, ey+eh), (0, 255,    0), 2)
    return frame

#giriş

reader = imageio.get_reader('1.mp4')
fps =   reader.get_meta_data()['fps']
writer  =   imageio.get_writer('output.mp4',    fps=fps)

#reader ile alınan videoyu döndürmek için for döngüsü
#enumarete fonksiyonu her frame bir sayı atıyor
for i,  frame in enumerate(reader):
    frame = detect(frame)
    writer.append_data(frame)
    print(i)

writer.close()



