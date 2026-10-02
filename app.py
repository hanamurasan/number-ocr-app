import io,re,cv2,numpy as np,pandas as pd,streamlit as st
from PIL import Image
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string,column_index_from_string,get_column_letter
B=Path(__file__).parent;st.set_page_config(page_title='一桁ずつ認識→Excel',page_icon='🔢',layout='wide');st.title('🔢 数字を一桁ずつ認識してExcelへ入力');st.caption('十字を自動検出した後、各桁を0〜9から個別判定します。登録済みの数値全体から選ぶ方式ではありません。')
L=pd.read_csv(B/'labels.csv',dtype=str);CT=[cv2.imread(str(p),0) for p in (B/'cross_templates').glob('*.png')]
def timekey(n):
 m=re.search(r'_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$',n,re.I);return (0,float(m.group(1))) if m else (1,n.lower())
def cross(rgb):
 g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);e=cv2.Canny(g,45,140);h,w=g.shape;best=(-2,None,None)
 for t0 in CT:
  z0=cv2.Canny(t0,45,140)
  for s in np.linspace(.45,2.4,32):
   tw,th=max(10,int(z0.shape[1]*s)),max(10,int(z0.shape[0]*s))
   if tw>=w or th>=h:continue
   z=cv2.resize(z0,(tw,th));r=cv2.matchTemplate(e,z,cv2.TM_CCOEFF_NORMED);mask=np.full(r.shape,-2,np.float32);mask[int(r.shape[0]*.12):int(r.shape[0]*.9),int(r.shape[1]*.1):int(r.shape[1]*.9)]=r[int(r.shape[0]*.12):int(r.shape[0]*.9),int(r.shape[1]*.1):int(r.shape[1]*.9)];_,sc,_,loc=cv2.minMaxLoc(mask)
   if sc>best[0]:best=(sc,(loc[0]+tw//2,loc[1]+th//2),max(tw,th))
 return best
def roi(rgb,c,sz):
 cx,cy=c;u=sz/30.;h,w=rgb.shape[:2];box=(max(0,int(cx-45*u)),max(0,int(cy-30*u)),min(w,int(cx-3*u)),min(h,int(cy-6*u)));return rgb[box[1]:box[3],box[0]:box[2]],box
def normroi(x):return cv2.resize(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY),(126,72),interpolation=cv2.INTER_CUBIC)
# Three fixed character slots inside cross-normalized ROI: tens, ones, tenths. Decimal is inserted by rule.
S=[(0,42),(32,74),(80,122)]
def charfeat(g,i):
 x1,x2=S[i];p=g[5:70,x1:x2];p=cv2.resize(p,(32,48));p=cv2.Laplacian(p,cv2.CV_32F);return (p-p.mean())/(p.std()+1e-6)
T=[]
for _,r in L.iterrows():
 rgb=np.array(Image.open(B/'training'/r.filename).convert('RGB'));sc,c,sz=cross(rgb)
 if c:
  rr,_=roi(rgb,c,sz);g=normroi(rr)
  for i,ch in enumerate(r.value.replace('.','')):T.append((i,ch,charfeat(g,i)))
def recognize(rgb):
 sc,c,sz=cross(rgb)
 if not c:return '',0,None,None,[]
 rr,box=roi(rgb,c,sz);g=normroi(rr);out='';details=[];scores=[]
 for i in range(3):
  q=charfeat(g,i);rank=sorted([(float((q*t).mean()),d) for pos,d,t in T if pos==i],reverse=True);out+=rank[0][1];scores.append(rank[0][0]);details.append(rank[:4])
 return out[:2]+'.'+out[2],float(np.mean(scores)),box,c,details
def cellok(x):return bool(re.fullmatch(r'[A-Za-z]{1,3}[1-9][0-9]*',x.strip()))
excel=st.file_uploader('1. 入力先Excel',type=['xlsx']);fs=sorted(st.file_uploader('2. 写真を選択',type=['png','jpg','jpeg','webp'],accept_multiple_files=True) or [],key=lambda f:timekey(f.name))
if fs:st.info('時間順：'+' → '.join(x.name for x in fs))
rows=[]
for i,f in enumerate(fs):
 rgb=np.array(Image.open(f).convert('RGB'));v,conf,box,c,det=recognize(rgb);m=rgb.copy()
 if box:cv2.rectangle(m,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
 if c:cv2.drawMarker(m,c,(255,0,255),cv2.MARKER_CROSS,20,2)
 a,b=st.columns([1,2]);a.image(m,caption=f.name);v=b.text_input('認識結果',v,key=str(i)+f.name);b.write(f'3桁の平均類似度 {conf:.2f}')
 if conf<.45:b.warning('信頼度が低いため確認してください。')
 with b.expander('各桁の候補'):
  for k,x in enumerate(det):b.write(f'{k+1}桁目：'+', '.join(f'{d}({s:.2f})' for s,d in x))
 rows.append({'filename':f.name,'value':v,'confidence':round(conf,3)})
if rows:
 st.dataframe(pd.DataFrame(rows),width='stretch')
 if excel:
  wb=load_workbook(io.BytesIO(excel.getvalue()));sn=st.selectbox('入力シート',wb.sheetnames);start=st.text_input('開始セル','C4')
  if cellok(start):
   letters,r0=coordinate_from_string(start.upper());col=column_index_from_string(letters);ws=wb[sn]
   for j,x in enumerate(rows):ws.cell(r0+j,col,float(x['value']));ws.cell(r0+j,col).number_format='0.0'
   end=f'{get_column_letter(col)}{r0+len(rows)-1}';o=io.BytesIO();wb.save(o);st.success(f'{start.upper()}:{end}へ入力しました');st.download_button('入力済みExcelをダウンロード',o.getvalue(),f'{Path(excel.name).stem}_入力済み.xlsx')
