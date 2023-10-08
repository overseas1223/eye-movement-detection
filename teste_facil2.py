from pickle import TRUE
from queue import Empty
import cv2 as cv
from matplotlib.animation import ImageMagickFileWriter
import mediapipe as mp
import time
import timeit
import utils, math
import numpy as np
from matplotlib import pyplot as plt
import uvicorn
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer

web = WebGear(logging=True)

# declaracao de variaveis
contador_frame =0
contador_direita=0
contador_esquerda =0
contador_centro =0 
contador_fechado =0
ond = 0

global i , j
i=0
j=0

global posicoes_d, posicoes_e, resultpc, resulteu, auxeu, auxpc
posicoes_d = ['CENTRO']
posicoes_e = ['CENTRO']
auxpc = list()
resultpc = list()
auxeu = list()
resulteu = list()


FONTS = cv.FONT_HERSHEY_COMPLEX


# identificacao dos pontos de referencia
FACE_OVAL           = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
LABIO               = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LABIO_INFERIOR      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LABIO_SUPERIOR      = [185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
OLHO_ESQUERDO       = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
SOBRANC_ESQUERDA    = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
OLHO_DIREITO        = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]  
SOBRANC_DIREITA     = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

mapa_face_mesh = mp.solutions.face_mesh


# identificacao da camera

async def initCamera():
    camera = cv.VideoCapture(0) # "todos.webm", t118.mp4 ou "t039.mp4"  "facil.webm") ( para camera ao vivo deixe "0")
    _, frame = camera.read()
    img = cv.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv.INTER_CUBIC)
    img_altura, img_largura = img.shape[:2]
    print(img_altura, img_largura)
    return img_altura,img_largura


### teste time 
#start_time = time.time()

def tic():
    global start_time 
    start_time = time.time()

def tac1():
    t_sec = round(time.time() - start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    #(t_hour,t_min) = divmod(t_min,60) 
    print(' {}min:{}sec'.format(t_min,t_sec)) #(t_hour,t_min,t_sec)
    auxpc.append("{}min:{}sec".format(t_min,t_sec))


def tac2():
    global t2
    global resulteu
    t_sec = round(time.time() - start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    #(t_hour,t_min) = divmod(t_min,60) 
    print(' {}min:{}sec'.format(t_min,t_sec)) #(t_hour,t_min,t_sec)
    auxeu.append("{}min:{}sec".format(t_min,t_sec))


# funcao de identificacao dos pontos (landmark detection) 
def landmarksDetection(img, resultados, draw=False):
    img_altura, img_largura= img.shape[:2]
    mesh_coord = [(int(point.x * img_largura), int(point.y * img_altura)) for point in resultados.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
 
    # retorna a lista de tuplas para cada landmark
    return mesh_coord


# Função de extratação de olhos
def extracaoOlhos(img, olho_direito_coords, olho_esquerdo_coords):
    # converte imagem para escala de cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # obtem a dimensão da imagem
    dim = gray.shape
    
    # cria máscara a partir da dimensão da imagem em escala de cinza
    mask = np.zeros(dim, dtype=np.uint8)

    # desenha a forma dos olhos como uma máscara de cor branca
    cv.fillPoly(mask, [np.array(olho_direito_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(olho_esquerdo_coords, dtype=np.int32)], 255)

    # exibe a máscara 
    #cv.imshow('mask', mask)
    
    # desenha os olhos dentro da forma da mascara
    olhos = cv.bitwise_and(gray, gray, mask=mask)
    # muda a cor preta para cinza 
    #cv.imshow('olhos draw', olhos)
    olhos[mask==0]=155
    
    # obtem (x,y) mínimos e máximos para os olhos direito e esquerdo 
    # olho direito
    r_max_x = (max(olho_direito_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(olho_direito_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(olho_direito_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(olho_direito_coords, key=lambda item: item[1]))[1]

    # olho esquerdo
    l_max_x = (max(olho_esquerdo_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(olho_esquerdo_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(olho_esquerdo_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(olho_esquerdo_coords, key=lambda item: item[1]))[1]

    # recorta os olhos da máscara
    recorte_direito = olhos[r_min_y: r_max_y, r_min_x: r_max_x]
    recorte_esquerdo = olhos[l_min_y: l_max_y, l_min_x: l_max_x]

    # retorna os olhos recortados 
    return recorte_direito, recorte_esquerdo
    

# estima o nivel de luz do video
def nivelluz(img):
    #blur = cv.blur(img, (9, 9))
    mean = np.mean(img)
    #print(mean)
    if mean > 130:
        thresh = 50
    elif mean < 130 and mean > 100:
        thresh = 40
    elif mean < 100 and mean > 70:
        thresh = 30
    else:
        thresh = 25
    return thresh


# Estimador de posição dos olhos
def estimadorPosicao(olhos_recortados_d, olhos_recortados_e):
    global i
    global j
    global posicoes_d
    global posicoes_e
 
    # obtem altura e largura dos olhos 
    h, w = olhos_recortados_d.shape
    y, x = olhos_recortados_e.shape
    #print(h, w)

    # remove o ruido das imagens
    gaussain_blur_d = cv.GaussianBlur(olhos_recortados_d, (9,9),0)
    gaussain_blur_e = cv.GaussianBlur(olhos_recortados_e, (9,9),0)
    #cv.imshow('gau', gaussain_blur)

    median_blur_d = cv.medianBlur(gaussain_blur_d, 3)
    median_blur_e = cv.medianBlur(gaussain_blur_e, 3)
    #cv.imshow('med', median_blurd)


    ### obtem nivel de luz da imagem
    threshd = nivelluz(median_blur_d)
    threshe = nivelluz(median_blur_e)


    # aplica thrsholding para converter  para imagem binária 
    ret, threshed_eye_d = cv.threshold(median_blur_d, threshd, 255, cv.THRESH_BINARY)
    ret, threshed_eye_e = cv.threshold(median_blur_e, threshe, 255, cv.THRESH_BINARY) #median_blure ou olhos_recortados_e, 130, 255, cv.THRESH_BINARY)
    #cv.imshow("thres", threshed_eyed)

   
    p = 0
    q = 0

    # para olho direito
    contours_d, hierarchy_d = cv.findContours(threshed_eye_d,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for c in contours_d:
    # calculate moments for each contour
        p = p+1
        M = cv.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
         
        # display the image
        cv.circle(olhos_recortados_d, (cX, cY), 2, (255, 255, 255), -1)
        cv.putText(olhos_recortados_d, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #cv.imshow("teste1", olhos_recortados_d) #threshed_eyed)
        #cv.waitKey(0)
        
        if p==1:
            d1 = cX
            #print("p", p, "x1", cX)
        elif p==2:
            d2 = cX
            #print("p", p, "x1", cX)


    if p==2:
        distancia_d = d1-d2
        if distancia_d > 3:
            i += 1
            color=[utils.BLACK, utils.GREEN]
            posicoes_d.append('DIREITA')  
        elif distancia_d >= -3 and distancia_d <= 3:
            i += 1
            color = [utils.YELLOW, utils.PINK]
            posicoes_d.append('CENTRO')  
        elif distancia_d < -3:
            i += 1
            color = [utils.GRAY, utils.YELLOW]
            posicoes_d.append('ESQUERDA')  
    elif p==1:
        i += 1
        posiantiga = posicoes_d[i-1]
        posicoes_d.append(posiantiga)
    elif p > 2:
        i += 1
        posiantiga = posicoes_d[i-1]
        posicoes_d.append(posiantiga)
        

    # para olho esquerdo
    contours_e, hierarchy_e = cv.findContours(threshed_eye_e,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for k in contours_e:
    # calculate moments for each contour
        q = q+1
        N = cv.moments(k)
        # calculate x,y coordinate of center
        if (int(N["m00"]))==0:
            kX = 0
            kY = 0
        else: 
            kX = int(N["m10"] / N["m00"])
            kY = int(N["m01"] / N["m00"])
                
        # display the image
        cv.circle(threshed_eye_e, (kX, kY), 2, (255, 255, 255), -1)
        cv.putText(threshed_eye_e, "centroid", (kX - 25, kY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #cv.imshow("teste2", threshed_eye_e) 
        #cv.waitKey(10)
        
        if q==1:
            e1 = kX
            #print("p", p, "x1", cX)
        elif q==2:
            e2 = kX
            #print("p", p, "x1", cX)


    if q==2:
        distancia_e = e1-e2
        if distancia_e > 3:
            j += 1
            color=[utils.BLACK, utils.GREEN]
            posicoes_e.append('DIREITA') 
        elif distancia_e >= -3 and distancia_e <= 3:
            j += 1
            color = [utils.YELLOW, utils.PINK]
            posicoes_e.append('CENTRO')
        elif distancia_e < -3:
            j += 1
            color = [utils.GRAY, utils.YELLOW]
            posicoes_e.append('ESQUERDA')
    elif q==1:
        j += 1
        posiantiga = posicoes_e[j-1]
        posicoes_e.append(posiantiga)
    elif q > 2:
        j += 1
        posiantiga = posicoes_e[j-1]
        posicoes_e.append(posiantiga)
        
  ##### testar o fechado
    if  h < 9 and y < 9:
        posicao_olho_d = "FECHADO"
        posicao_olho_e = "FECHADO"
    else: 
        posicao_olho_d = posicoes_d[i]
        posicao_olho_e = posicoes_e[j]


    return posicao_olho_d, posicao_olho_e 


with mapa_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh:

    #start_time = time.time()
    tic()
 
    while True:
        contador_frame +=1 
        ret, frame = camera.read()
        if not ret: 
            break 
        
        frame = cv.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv.INTER_CUBIC)
        frame_altura, frame_largura= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        resultados  = face_mesh.process(rgb_frame)

        if resultados.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, resultados, False)
        
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in OLHO_ESQUERDO ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in OLHO_DIREITO], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
        
            right_coords = [mesh_coords[p] for p in OLHO_DIREITO]
            left_coords = [mesh_coords[p] for p in OLHO_ESQUERDO]

            recorte_direita, recorte_esquerda = extracaoOlhos(frame, right_coords, left_coords)
            #cv.imshow('right', recorte_direita) #mostra olho direito
            #cv.imshow('left', recorte_esquerda) #mostra olho esquerdo

            posicao_olhos_direita, posicao_olhos_esquerda = estimadorPosicao(recorte_direita, recorte_esquerda)
                 
            
            if (posicao_olhos_direita=="DIREITA" and posicao_olhos_esquerda=="DIREITA") and contador_direita<2:
                #start = time.process_time()
                contador_direita+=1
                contador_centro=0
                contador_esquerda=0
                contador_fechado=0
 
                auxpc.append("DIREITA")
                print("Direita:")
                tac1()
                resultpc.append(auxpc[:])
                auxpc.clear()


            if (posicao_olhos_direita=="CENTRO" or posicao_olhos_esquerda=="CENTRO") and contador_centro <2:
                #start = time.process_time()
                contador_centro +=1
                contador_direita=0
                contador_esquerda=0
                contador_fechado=0
                
                auxpc.append("CENTRO")
                print("Centro:")
                tac1()
                resultpc.append(auxpc[:])
                auxpc.clear()               
            
            if (posicao_olhos_direita=="ESQUERDA" and posicao_olhos_esquerda=="ESQUERDA") and contador_esquerda<2: 
                #start = time.process_time()
                contador_esquerda +=1
                contador_centro=0
                contador_direita=0
                contador_fechado=0
                
                auxpc.append("ESQUERDA")
                print("Esquerda:")
                tac1()
                resultpc.append(auxpc[:])
                auxpc.clear()  

            if (posicao_olhos_direita=="FECHADO"  or posicao_olhos_esquerda=="FECHADO") and contador_fechado<2: 
                #start = time.process_time()
                contador_esquerda=0
                contador_centro=0
                contador_direita=0
                contador_fechado +=1
                
                auxpc.append("FECHADO")
                print("Fechado:")
                tac1()
                resultpc.append(auxpc[:])
                auxpc.clear()
               
        else:
            print("olhos não detectados")
            tac1()
            
  
        end_time = time.time()-start_time
        fps = contador_frame/end_time
        #print("fps", fps)    

        cv.imshow('frame', frame)
        #esse waitkey() define a velocidade dos frames
        key = cv.waitKey(50)
        
        #para controle manual
        if key==ord('j'):   
            auxeu.append("DIREITA")
            print("A - DIREITA")
            tac2()
            resulteu.append(auxeu[:])
            auxeu.clear()
        if key==ord('k'):
            auxeu.append("CENTRO")
            print("A - CENTRO")
            tac2()
            resulteu.append(auxeu[:])
            auxeu.clear()
        if key==ord('l'):
            auxeu.append("ESQUERDA")
            print("A - ESQUERDA")
            tac2()
            resulteu.append(auxeu[:])
            auxeu.clear()  
            
        #para fechar o programa  
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()
    print("Fim do vídeo")

matrizeu = np.asarray(resulteu)
matrizpc = np.asarray(resultpc)
np.savetxt("resultado_eu_facil.csv", matrizeu, delimiter=" ; ", fmt='%s')
np.savetxt("resultado_pc_facil.csv", matrizpc, delimiter=" ; ", fmt='%s')
print("Resultados gerados")

web.config["generator"] = my_frame_producer

    # run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

    # close app safely
web.shutdown()
