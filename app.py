import streamlit as st
import pandas as pd
from PIL import Image
import torch
from joblib import load
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
from streamlit.components.v1 import html



# عنوان التطبيق
st.title("نظام تحليل الصور وتقدير العمر والجنس")

# تحميل بيانات CSV
def load_data():
    try:
        data = pd.read_csv('recom.csv', encoding='utf-8')
        return data
    except FileNotFoundError:
        st.warning("ملف البيانات  غير موجود")
        return pd.DataFrame()

data = load_data()

# تحميل نماذج التنبؤ
@st.cache_resource
def load_models():
    try:
      # Load the processor and models first
        processor = AutoImageProcessor.from_pretrained("dima806/fairface_age_image_detection")
        age_model = AutoModelForImageClassification.from_pretrained("dima806/fairface_age_image_detection")
        gender_model = AutoModelForImageClassification.from_pretrained("dima806/fairface_gender_image_detection")
        return processor, age_model, gender_model
    except Exception as e:
        st.error(f"حدث خطأ في تحميل النماذج: {e}")
        return None, None, None

processor, age_model, gender_model = load_models()

# وظيفة للتنبؤ بالعمر والجنس
def predict_image(image):
    try:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            age_logits = age_model(**inputs).logits
            gender_logits = gender_model(**inputs).logits

            # Get predictions
            predicted_age_idx = age_logits.argmax(-1).item()
            predicted_gender_idx = gender_logits.argmax(-1).item()

            # Decode predictions
            predicted_age = age_model.config.id2label[predicted_age_idx]
            predicted_gender = gender_model.config.id2label[predicted_gender_idx]
            
            return predicted_age, predicted_gender
    except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {e}")
        return None, None

def cards(recommendations):
    # Custom CSS for the cards
    css = """
    <style>
    body {
     margin: 0;
    padding: 0;
    
    }
     [data-testid="stAppViewContainer"] ,section{
        background-color: #3559A0;
    }
    .flex-container {
        display: flex;
        flex-wrap: nowrap;
        gap: 15px;
        justify-content: flex-start;
        direction: rtl;
        padding: 20px;
background-color: #22305C;  
        overflow-x: auto;
        scrollbar-color: #3559A0 #22305C;
        scrollbar-width: thin;
         box-shadow: 0 2px 8px rgba(0,0,0,0.07);


  }
  iframe,{
          border: 1px solid #3559A0;
        border-radius: 20px;
        background-color: #22305C;  

        }
    .card {
        background: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 16px;
        width: 220px;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 15px;
         flex: 0 0 auto;
    }
    .card img {
        width: 100%;
        height: 160px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #eee;
    }
    .card h4 {
        margin: 8px 0;
        color: #333;
        font-size: 16px;
    }
    .card p {
        margin: 4px 0;
        color: #555;
        font-size: 14px;
    }
    </style>
    """
    
    # Create a complete HTML document
    cards_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        {css}
    </head>
    <body>
        <div class="flex-container">
    """
    
    # Build the flex cards
    for _, row in recommendations.iterrows():
        # Handle missing image case
        image_url = row.get('image', '') if pd.notna(row.get('image', '')) else "https://via.placeholder.com/220x160?text=No+Image"
        
        cards_html += f"""
        <div class="card">
            <img src="{image_url}" alt="{row.get('name', '')}" onerror="this.src='https://via.placeholder.com/220x160?text=Image+Error'"/>
            <h4>{row.get('name', '')}</h4>
            <p>{row.get('type', '')} | {row.get('genre', '')}</p>
            <p>العمر: {row.get('age_group', '')} | الجنس: {row.get('gender', '')}</p>
        </div>
        """
    
    cards_html += """
        </div>
    </body>
    </html>
    """
    
    # Use Streamlit's html component to render the HTML properly
    html(cards_html, height=320, scrolling=False)


# وظيفة لعرض التوصيات بناءً على العمر والجنس
def show_recommendations(age, gender, data):
    if data.empty:
        st.warning("لا توجد بيانات توصيات متاحة")
        return
    
    # فلترة البيانات بناءً على العمر والجنس (يمكن تعديل هذا المنطق حسب احتياجاتك)
    try:
        # تحويل العمر إلى رقم للتصفية
        
        # تصفية حسب الجنس
        
        # يمكنك تعديل منطق التوصية هنا حسب عمود العمر في بياناتك
        recommendations = data.loc[
            (data['gender'] == gender) & 
            (data['age_group'] ==age) 
        ]
        
        st.subheader("التوصيات المقترحة:")
        
        if not recommendations.empty:
            # عرض التوصيات باستخدام بطاقات
            cards(recommendations)  
        else: 
            st.warning(recommendations)    
    except Exception as e:
        st.error(f"حدث خطأ في عرض التوصيات: {e}")
        st.dataframe(data.sample(5))

# إنشاء قائمة جانبية للاختيارات
option = st.sidebar.selectbox(
    "اختر طريقة إدخال الصورة",
    ("تحميل من الملف", "التقاط من الكاميرا")
)

# متغير للصورة
uploaded_image = None
captured_image = None
image_to_predict = None

if option == "تحميل من الملف":
    uploaded_file = st.file_uploader("اختر صورة لتحميلها", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # عرض شاشة التحميل
        with st.spinner('جاري معالجة الصورة...'):
            uploaded_image = Image.open(uploaded_file)
            image_to_predict = uploaded_image
            st.success("تم تحميل الصورة بنجاح!")
            
            # عرض الصورة
            st.image(uploaded_image, caption="الصورة المرفوعة", use_column_width=True)

else:
    # خيار التقاط صورة من الكاميرا
    st.write("اضغط على الزر لتفعيل الكاميرا")
    
    picture = st.camera_input("التقاط صورة")
    
    if picture:
        with st.spinner('جاري معالجة الصورة...'):
            captured_image = Image.open(picture)
            image_to_predict = captured_image
            st.success("تم التقاط الصورة بنجاح!")

# زر لمعالجة الصورة واستخراج البيانات
if st.button("تحليل الصورة"):
    if image_to_predict is not None and processor is not None and age_model is not None and gender_model is not None:
        with st.spinner('جاري تحليل الصورة...'):
            predicted_age, predicted_gender = predict_image(image_to_predict)
            
            if predicted_age is not None and predicted_gender is not None:
                # عرض النتائج
                st.subheader("نتائج التحليل:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("العمر المتوقع", predicted_age)
                with col2:
                    st.metric("الجنس المتوقع",predicted_gender)
                
                # عرض التوصيات
                show_recommendations(predicted_age.lower().strip(), predicted_gender.lower().strip(), data)
            else:
                st.error("فشل في تحليل الصورة")
    else:
        if image_to_predict is None:
            st.warning("الرجاء تحميل أو التقاط صورة أولاً")
        else:
            st.error("النماذج غير جاهزة للتحليل")

# قسم لإضافة ملف CSV جديد إذا لزم الأمر
st.sidebar.header("إدارة البيانات")
new_csv = st.sidebar.file_uploader("رفع ملف بيانات جديد (CSV)", type=['csv'])
if new_csv is not None:
    try:
        new_data = pd.read_csv(new_csv, encoding='utf-8')
        new_data.to_csv('data.csv', index=False)
        st.sidebar.success("تم تحديث بيانات CSV بنجاح!")
        data = load_data()  # إعادة تحميل البيانات
    except Exception as e:
        st.sidebar.error(f"حدث خطأ: {e}")