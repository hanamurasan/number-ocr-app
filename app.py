import os,cv2,numpy as np,pandas as pd,streamlit as st
from PIL import Image
from pathlib import Path
BASE=Path(__file__).parent
st.set_page_config(page_title='専用数値認識 v4',page_icon='🔢',layout='wide')
st.title('🔢 この画面専用の数値認識 v4')
st.caption('登録画像から数字の形を比較します。Tesseractは使用しません。位置調整も不要です。')
labels=pd.read_csv(BASE/'labels.csv', dtype={'filename': str, 'value': str})
SLOTS=[(98,106),(105,113),(115,123)]
def normalized_gray(rgb):
 return cv2.resize(rgb,(237,228),interpolation=cv2.INTER_AREA)
def patch(rgb,slot):
 gray=cv2.cvtColor(normalized_gray(rgb),cv2.COLOR_RGB2GRAY);x1,x2=SLOTS[slot]
 p=gray[117:133,x1:x2];p=cv2.resize(p,(24,40),interpolation=cv2.INTER_CUBIC)
 p=cv2.Laplacian(p,cv2.CV_32F);return (p-p.mean())/(p.std()+1e-6)
def display_feature(rgb):
 gray=cv2.cvtColor(normalized_gray(rgb),cv2.COLOR_RGB2GRAY)
 p=gray[115:134,96:124];p=cv2.resize(p,(168,114),interpolation=cv2.INTER_CUBIC)
 p=cv2.Laplacian(p,cv2.CV_32F);return (p-p.mean())/(p.std()+1e-6)
training=[];chars=[]
for _,r in labels.iterrows():
 filename=str(r['filename']); value=str(r['value'])
 rgb=np.array(Image.open(BASE/'training'/filename).convert('RGB')); training.append((value,display_feature(rgb)))
 for i,ch in enumerate(value.replace('.','')):chars.append((ch,i,patch(rgb,i)))
def recognize(rgb):
 q=display_feature(rgb); full=sorted([(float((q*t).mean()),v) for v,t in training],reverse=True)
 pred='';cs=[]
 for i in range(3):
  z=patch(rgb,i); ranked=sorted([(float((z*t).mean()),ch) for ch,pos,t in chars if pos==i],reverse=True)
  pred+=ranked[0][1];cs.append(ranked[0][0])
 char_value=pred[:2]+'.'+pred[2]
 # Practically identical known display gets the full-template result; otherwise character templates generalize.
 if full[0][0]>=0.82: value=full[0][1]; method='登録済み表示との全体比較'; conf=full[0][0]
 else: value=char_value; method='文字ごとの形状比較'; conf=float(np.mean(cs))
 return value,conf,method,full[:5],char_value
files=st.file_uploader('画像を選択（複数可）',type=['png','jpg','jpeg','webp'],accept_multiple_files=True)
rows=[]
for i,f in enumerate(files or []):
 rgb=np.array(Image.open(f).convert('RGB')); norm=normalized_gray(rgb)
 value,conf,method,near,char_value=recognize(rgb)
 marked=norm.copy();cv2.rectangle(marked,(96,115),(124,134),(0,255,0),2)
 a,b=st.columns(2);a.image(rgb,caption='元画像');b.image(marked,caption='自動認識範囲')
 corrected=st.text_input('認識結果',value,key=str(i)+f.name)
 st.write(f'方式: **{method}**　類似度: **{conf:.2f}**')
 if conf<0.55: st.warning('見本との類似度が低いため、結果を確認してください。')
 with st.expander('判定の詳細'):
  st.write('文字別候補:',char_value);st.dataframe(pd.DataFrame(near,columns=['類似度','登録値']))
 rows.append({'filename':f.name,'value':corrected,'similarity':round(conf,3),'method':method})
 st.divider()
if rows:
 df=pd.DataFrame(rows);st.dataframe(df,use_container_width=True)
 st.download_button('CSVをダウンロード',df.to_csv(index=False).encode('utf-8-sig'),'ocr_results.csv','text/csv')
