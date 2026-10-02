import io,cv2,numpy as np,pandas as pd,streamlit as st
from PIL import Image
from pathlib import Path
from openpyxl import load_workbook
BASE=Path(__file__).parent
st.set_page_config(page_title='画像→Excel 数値入力',page_icon='📊',layout='wide')
st.title('📊 画像から数値を読み取りExcelへ入力')
st.caption('複数画像を読み取り、選択したExcelのC4から縦方向に書き込みます。')
labels=pd.read_csv(BASE/'labels.csv',dtype=str); SLOTS=[(98,106),(105,113),(115,123)]
def norm(rgb): return cv2.resize(rgb,(237,228),interpolation=cv2.INTER_AREA)
def patch(rgb,i):
 g=cv2.cvtColor(norm(rgb),cv2.COLOR_RGB2GRAY);x1,x2=SLOTS[i];p=cv2.resize(g[117:133,x1:x2],(24,40),interpolation=cv2.INTER_CUBIC);p=cv2.Laplacian(p,cv2.CV_32F);return (p-p.mean())/(p.std()+1e-6)
def whole(rgb):
 g=cv2.cvtColor(norm(rgb),cv2.COLOR_RGB2GRAY);p=cv2.resize(g[115:134,96:124],(168,114),interpolation=cv2.INTER_CUBIC);p=cv2.Laplacian(p,cv2.CV_32F);return (p-p.mean())/(p.std()+1e-6)
training=[];chars=[]
for _,r in labels.iterrows():
 value=str(r['value']);rgb=np.array(Image.open(BASE/'training'/str(r['filename'])).convert('RGB'));training.append((value,whole(rgb)))
 for i,ch in enumerate(value.replace('.','')):chars.append((ch,i,patch(rgb,i)))
def recognize(rgb):
 q=whole(rgb);full=sorted([(float((q*t).mean()),v) for v,t in training],reverse=True);digits='';scores=[]
 for i in range(3):
  z=patch(rgb,i);best=max((float((z*t).mean()),ch) for ch,pos,t in chars if pos==i);scores.append(best[0]);digits+=best[1]
 charval=digits[:2]+'.'+digits[2]
 return (full[0][1],full[0][0],'登録表示との全体比較') if full[0][0]>=.82 else (charval,float(np.mean(scores)),'文字別比較')
excel=st.file_uploader('1. 入力先のExcelファイルを選択（.xlsx）',type=['xlsx'])
images=st.file_uploader('2. 数値画像を入力順に選択（複数可）',type=['png','jpg','jpeg','webp'],accept_multiple_files=True)
rows=[]
for i,f in enumerate(images or []):
 rgb=np.array(Image.open(f).convert('RGB'));value,score,method=recognize(rgb)
 c1,c2=st.columns([1,2]);c1.image(rgb,caption=f.name);corrected=c2.text_input('認識結果',value,key=str(i)+f.name);c2.write(f'類似度 {score:.2f} / {method}')
 rows.append({'filename':f.name,'value':corrected})
if rows:
 st.subheader('3. 認識結果');df=pd.DataFrame(rows);st.dataframe(df,use_container_width=True)
 if excel:
  try:
   wb=load_workbook(io.BytesIO(excel.getvalue()));sheet=st.selectbox('入力するシート',wb.sheetnames);ws=wb[sheet]
   for r,item in enumerate(rows,start=4):
    try: ws.cell(r,3,float(item['value']))
    except: ws.cell(r,3,item['value'])
    ws.cell(r,3).number_format='0.0'
   out=io.BytesIO();wb.save(out)
   st.success(f'{sheet} の C4:C{len(rows)+3} に入力しました。')
   st.download_button('4. 入力済みExcelをダウンロード',out.getvalue(),f'{Path(excel.name).stem}_入力済み.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
  except Exception as e: st.error(f'Excelを処理できませんでした: {e}')
 else: st.info('Excelファイルも選択すると、C4から入力したファイルを作成できます。')
