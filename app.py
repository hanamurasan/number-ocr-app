import io,re,cv2,numpy as np,pandas as pd,streamlit as st
from PIL import Image
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string,column_index_from_string,get_column_letter
BASE=Path(__file__).parent
st.set_page_config(page_title='自動検出 数値→Excel',page_icon='🎯',layout='wide')
st.title('🎯 数字位置を自動検出してExcelへ入力')
st.caption('写真の縦横比や撮影距離が変わっても、十字マークを探して数字領域を自動切り出しします。')
labels=pd.read_csv(BASE/'labels.csv',dtype=str)
CROSS=[cv2.imread(str(p),0) for p in sorted((BASE/'cross_templates').glob('*.png'))]
def time_key(name):
 m=re.search(r'_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$',name,re.I)
 return (0,float(m.group(1))) if m else (1,name.lower())
def detect_cross(rgb):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);edge=cv2.Canny(gray,45,140);h,w=gray.shape;best=(-2,None,None)
 for templ0 in CROSS:
  te0=cv2.Canny(templ0,45,140)
  for s in np.linspace(.45,2.4,40):
   tw=max(10,int(te0.shape[1]*s));th=max(10,int(te0.shape[0]*s))
   if tw>=w or th>=h:continue
   te=cv2.resize(te0,(tw,th),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC)
   res=cv2.matchTemplate(edge,te,cv2.TM_CCOEFF_NORMED)
   # Crosshair is normally inside the central 80%; suppress device logos and borders.
   mask=np.full(res.shape,-2,np.float32);xa,xb=int(res.shape[1]*.1),int(res.shape[1]*.9);ya,yb=int(res.shape[0]*.12),int(res.shape[0]*.9);mask[ya:yb,xa:xb]=res[ya:yb,xa:xb]
   _,score,_,loc=cv2.minMaxLoc(mask)
   if score>best[0]:best=(score,(loc[0]+tw//2,loc[1]+th//2),max(tw,th))
 return best
def roi_from_cross(rgb,c,size):
 cx,cy=c;u=size/30.0;h,w=rgb.shape[:2]
 x1=max(0,int(cx-45*u));x2=min(w,int(cx-3*u));y1=max(0,int(cy-30*u));y2=min(h,int(cy-6*u))
 return rgb[y1:y2,x1:x2],(x1,y1,x2,y2)
def feature(roi):
 g=cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY);g=cv2.resize(g,(126,72),interpolation=cv2.INTER_CUBIC);p=cv2.Laplacian(g,cv2.CV_32F);return (p-p.mean())/(p.std()+1e-6)
# Build normalized templates by detecting each training image the same way used for new images.
TRAIN=[]
for _,r in labels.iterrows():
 rgb=np.array(Image.open(BASE/'training'/r.filename).convert('RGB'));sc,c,sz=detect_cross(rgb)
 if c is not None:
  roi,_=roi_from_cross(rgb,c,sz);TRAIN.append((r.value,feature(roi)))
def recognize(rgb):
 score,c,size=detect_cross(rgb)
 if c is None:return '',0,None,None,[]
 roi,box=roi_from_cross(rgb,c,size);q=feature(roi);near=sorted([(float((q*t).mean()),v) for v,t in TRAIN],reverse=True)
 return near[0][1],near[0][0],box,c,near[:5]
def valid_cell(x):return bool(re.fullmatch(r'[A-Za-z]{1,3}[1-9][0-9]*',x.strip()))
excel=st.file_uploader('1. 入力先のExcel（.xlsx）',type=['xlsx'])
files=st.file_uploader('2. 写真をまとめて選択',type=['png','jpg','jpeg','webp'],accept_multiple_files=True)
files=sorted(files or [],key=lambda f:time_key(f.name))
if files:st.info('写真はファイル名末尾の秒数で昇順に並べました：'+ ' → '.join(f.name for f in files))
rows=[]
for i,f in enumerate(files):
 rgb=np.array(Image.open(f).convert('RGB'));val,sim,box,c,near=recognize(rgb);marked=rgb.copy()
 if box:cv2.rectangle(marked,(box[0],box[1]),(box[2],box[3]),(0,255,0),max(2,rgb.shape[1]//300))
 if c:cv2.drawMarker(marked,c,(255,0,255),cv2.MARKER_CROSS,max(18,rgb.shape[1]//18),2)
 a,b=st.columns([1,2]);a.image(marked,caption=f.name);corrected=b.text_input('認識結果',val,key=str(i)+f.name);b.write(f'類似度 {sim:.2f}')
 if sim<.45:b.warning('類似度が低いため、数値を確認してください。')
 with b.expander('候補'):b.dataframe(pd.DataFrame(near,columns=['類似度','値']))
 rows.append({'filename':f.name,'value':corrected,'similarity':round(sim,3)})
if rows:
 st.subheader('3. 結果確認');st.dataframe(pd.DataFrame(rows),use_container_width=True)
 if excel:
  wb=load_workbook(io.BytesIO(excel.getvalue()));sheet=st.selectbox('4. 入力シート',wb.sheetnames);start=st.text_input('5. 開始セル',value='C4')
  if valid_cell(start):
   letters,row0=coordinate_from_string(start.upper());col=column_index_from_string(letters);ws=wb[sheet]
   for j,x in enumerate(rows):
    cell=ws.cell(row0+j,col)
    try:cell.value=float(x['value'])
    except:cell.value=x['value']
    cell.number_format='0.0'
   end=f'{get_column_letter(col)}{row0+len(rows)-1}';out=io.BytesIO();wb.save(out)
   st.success(f'{sheet} の {start.upper()}:{end} に入力しました。')
   st.download_button('6. 入力済みExcelをダウンロード',out.getvalue(),f'{Path(excel.name).stem}_入力済み.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
  else:st.error('開始セルは C4 のように入力してください。')
