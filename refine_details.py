from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from reconstruct import reconstruct, ROOT

OUT=ROOT/'outputs/details'; OUT.mkdir(exist_ok=True)
v,meta=reconstruct(ROOT/'data/slices_step4')
smooth=ndi.gaussian_filter(v,.45)
level=float(threshold_otsu(smooth))
labels,_=ndi.label(smooth>level)
sizes=np.bincount(labels.ravel()); sizes[0]=0
component=labels==sizes.argmax()
roi=ndi.binary_dilation(component,iterations=1)
field=np.where(roi,smooth,0)

def build(field,name):
    vertices,faces,_,_=marching_cubes(np.pad(field,2),level=level,allow_degenerate=False)
    mesh=trimesh.Trimesh(vertices=vertices-2,faces=faces,process=True)
    if mesh.volume<0: mesh.invert()
    mesh.export(OUT/f'{name}.stl')
    loaded=trimesh.load_mesh(OUT/f'{name}.stl')
    assert loaded.is_watertight and loaded.is_winding_consistent and loaded.volume>0
    return loaded

whole=build(field,'brain_intensity_surface')
cut=field.copy(); cut[65:,:,:]=0
half=build(cut,'brain_cutaway')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
def draw(ax,mesh,azim):
    coll=Poly3DCollection(mesh.triangles,facecolors=np.where(((mesh.triangles[:,:,0].min(axis=1)>64) & (mesh is half))[:,None], np.array([.86,.68,.39,1]), np.array([.44,.71,.79,1])),shade=True,
                          lightsource=matplotlib.colors.LightSource(315,40))
    ax.add_collection3d(coll)
    center=np.array([64,62,64]); r=42
    ax.set(xlim=(center[0]-r,center[0]+r),ylim=(center[1]-r,center[1]+r),zlim=(center[2]-r,center[2]+r))
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=12,azim=azim);ax.set_axis_off()

fig=plt.figure(figsize=(15,5))
old=trimesh.load_mesh(ROOT/'outputs/embryonic_brain_envelope.stl')
for i,(m,title,az) in enumerate([(old,'Было: заполненная оболочка',15),(whole,'Теперь: границы ярких областей',15),(half,'Разрез: внутренние границы',10)]):
    ax=fig.add_subplot(1,3,i+1,projection='3d'); draw(ax,m,az);ax.set_title(title)
fig.suptitle('Те же 33 УЗИ-среза · без заполнения полостей',fontsize=19)
fig.text(.5,.035,'Золотистый участок — искусственная плоскость разреза. Контуры не являются экспертной разметкой.',ha='center')
fig.subplots_adjust(top=.84,bottom=.09,wspace=0);fig.savefig(OUT/'comparison.png',dpi=140);plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,x in zip(axs,[56,60,64]):
    ax.imshow(v[x,:,:].T,origin='lower',cmap='gray',vmin=0,vmax=.5)
    ax.contour(field[x,:,:].T,levels=[level],colors=['#52ebbd'],linewidths=.8)
    ax.set(xlim=(18,105),ylim=(22,105),title=f'Срез x={x} · контур поверхности');ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'contours.png',dpi=150);plt.close(fig)

stats={'input_planes':len(meta['files']),'smoothing_sigma_voxels':.45,'threshold':level,'closing':False,'fill_holes':False,'cut':'retain x indices 0..64; close interface between 64 and 65','units':'atlas grid units, not millimeters','models':{}}
for name,m in [('brain_intensity_surface',whole),('brain_cutaway',half)]:
    stats['models'][name]={'vertices':len(m.vertices),'faces':len(m.faces),'watertight':bool(m.is_watertight),'winding_consistent':bool(m.is_winding_consistent),'surface_components':len(m.split()),'volume_grid_units_cubed':float(m.volume)}
(OUT/'metrics.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
