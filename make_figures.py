"""Generate presentation figures from saved experiment outputs (Russian labels)."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs'
ref = nib.load(ROOT/'data/atlas_56.nii.gz').get_fdata()
rec = nib.load(OUT/'reconstructed_volume.nii.gz').get_fdata()
mask = nib.load(OUT/'envelope_mask.nii.gz').get_fdata()
mesh = trimesh.load_mesh(OUT/'embryonic_brain_envelope.stl')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})

def surface(ax, azim=35):
    coll = Poly3DCollection(mesh.triangles, facecolors='#75b6cc', shade=True,
                            lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45))
    ax.add_collection3d(coll)
    center=mesh.bounds.mean(axis=0); r=mesh.extents.max()/2*1.08
    ax.set(xlim=(center[0]-r,center[0]+r),ylim=(center[1]-r,center[1]+r),zlim=(center[2]-r,center[2]+r))
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=18,azim=azim); ax.set_axis_off()

fig=plt.figure(figsize=(15,5),facecolor='#f6f8fb')
for j,z in enumerate([44,64]):
    ax=fig.add_subplot(1,3,j+1)
    ax.imshow(rec[:,:,z].T,origin='lower',cmap='gray',vmin=0,vmax=.5)
    ax.contour(mask[:,:,z].T,levels=[.5],colors=['#6ee7ba'],linewidths=1)
    ax.set_title(f'Входной УЗИ-срез z={z}\nКонтур автоматической оболочки'); ax.axis('off')
ax=fig.add_subplot(1,3,3,projection='3d'); surface(ax)
ax.set_title('STL: оболочка области мозга\nМасштаб в единицах атласа')
fig.suptitle('2D-срезы УЗИ-атласа → 3D-оболочка',fontsize=20,fontweight='bold')
fig.text(.5,.025,'Контролируемый эксперимент • Усреднённый атлас, не отдельный плод • Не кожа и не модель лица',ha='center',fontsize=11)
fig.subplots_adjust(top=.80,bottom=.10,wspace=.04); fig.savefig(OUT/'result.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(12,8))
for row,z in enumerate([50,66]):
    for col,(arr,title) in enumerate([(ref,'Исходный отложенный срез'),(rec,'Линейная реконструкция'),(np.abs(rec-ref),'Абсолютная ошибка')]):
        im=axs[row,col].imshow(arr[:,:,z].T,origin='lower',cmap='magma' if col==2 else 'gray',vmin=0,vmax=.1 if col==2 else .5)
        axs[row,col].set_title(f'{title}\nz={z}, не передан на вход');axs[row,col].axis('off')
        if col==2: fig.colorbar(im,ax=axs[row,col],shrink=.75)
fig.suptitle('Проверка на срезах, отсутствующих во входном наборе',fontsize=16)
fig.tight_layout();fig.savefig(OUT/'validation.png',dpi=150);plt.close(fig)

fig=plt.figure(figsize=(12,4))
for i,az in enumerate([0,90,180]):
    ax=fig.add_subplot(1,3,i+1,projection='3d'); surface(ax,az);ax.set_title(f'Ракурс {az}°')
fig.tight_layout();fig.savefig(OUT/'model_views.png',dpi=150);plt.close(fig)
print('Saved result.png, validation.png, model_views.png')
