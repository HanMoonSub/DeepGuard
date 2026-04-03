import timm
from inference.image_predictor import ImagePredictor

# ----- 사용자 업로드 Image 딥페이크 여부 분석 ------
async def predict_image(image_loc: str, model_type: str, domain_type: str):
    
    print(f"model_type: {model_type}, domain_type: {domain_type}")
    
    model_name = None
    dataset = None

    # 사용자 지정 Model 불러오기
    if model_type == "fast":
        if domain_type == "서양인":
            model_name = "ms_eff_gcvit_b0"; dataset = "ff++"
        else:
            model_name = "ms_eff_gcvit_b0"; dataset = "kodf"
            
    else:
        if domain_type == "서양인":
            model_name = "ms_eff_gcvit_b5"; dataset = "ff++"
        else:
            model_name = "ms_eff_gcvit_b5"; dataset = "kodf"
            
    print(f"model_name: {model_name}, dataset: {dataset}")
            
    # Image Predictor 초기화 진행
    predictor = ImagePredictor(
            margin_ratio = 0.2, #  Margin ratio around the detected face crop
            conf_thres = 0.5, # Confidence threshold for face detection
            min_face_ratio = 0.01, # Minimum face-toframe size ratio to process 
            model_name = model_name,   
            dataset = dataset 
            )
    
    result = predictor.predict_img(
        img_path = f"./{image_loc}",
        tta_hflip = 0.0
    )
    
    print(f"Deepfake Probability: {result:.4f}")
    
    return result