"""Execute one isolated figure source and capture its actual saved output."""
from pathlib import Path
import sys,runpy,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
R=Path(__file__).resolve().parent
(R/'figures').mkdir(exist_ok=True)
name=sys.argv[1]; original=Figure.savefig; saved=[]; audit=[]
def save(fig,path,*args,**kwargs):
 # Each script has one canonical figure, even when it calls save twice for formats.
 stem=name
 if saved and id(fig)!=saved[-1]:stem=name+'_'+str(len(set(saved))+1)
 if id(fig) in saved:return
 saved.append(id(fig))
 # Rasterize dense map meshes, retaining vector labels.
 for ax in fig.axes:
  if hasattr(ax, 'projection'):
   for coll in ax.collections: coll.set_rasterized(True)
 # Remove standalone titles/footers, keeping panel labels and data legends.
 if fig._suptitle is not None:fig._suptitle.set_visible(False)
 for t in fig.texts:
  if t is not fig._suptitle and len(t.get_text())>100:t.set_visible(False)
 for ext in ['png','pdf']:
  original(fig,R/'figures'/f'{stem}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 widths=fig.get_size_inches()[0]
 fonts=[t.get_fontsize()*7.2/widths for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
 audit.append(dict(figure=stem,canvas_width_inches=float(widths),min_font_at_7_2_inches=min(fonts) if fonts else None))
Figure.savefig=save
plt.show=lambda *a,**k:None
runpy.run_path(str(R/(name+'.py')),run_name='__main__')
assert saved,'No figure saved'
(R/'figures'/f'{name}_font_audit.json').write_text(json.dumps(audit,indent=2))
