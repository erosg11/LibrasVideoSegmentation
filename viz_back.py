from fastapi import FastAPI, HTTPException, status, Depends, Query, Request
from pydantic import BaseModel
from fastapi.responses import ORJSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from orjson import loads
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Visualization Backend", description="API para vizualização de dados",
              default_response_class=ORJSONResponse, version='0.0.2')

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

LOCAL = Path('.')

labels_cefet: pd.DataFrame

labels_ufop: pd.DataFrame

project_label: Dict[str, pd.DataFrame]


class Results(BaseModel):
    results: List[str]


class VideoDataRecord(BaseModel):
    x: float
    y: float
    gesture_point: int
    critical: Optional[int]


class VideoData(BaseModel):
    data: List[VideoDataRecord]


@app.on_event('startup')
async def load_data():
    global labels_cefet, labels_ufop, project_label
    labels_cefet = pd.read_csv('labels - cefet.csv')
    labels_cefet.set_index('video', inplace=True)
    labels_ufop = pd.read_csv('labels.csv')
    labels_ufop = labels_ufop[['video', 'start', 'end']]
    labels_ufop.set_index('video', inplace=True)
    project_label = {
        'results_cefet': labels_cefet,
        'results': labels_ufop,
    }


@app.get('/results', response_model=Results)
async def get_results():
    return {'results': [x.stem for x in sorted(LOCAL.glob('result*')) if x.is_dir()]}


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


def get_sub_folders(base_folder: Path):
    if not base_folder.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Data no found')
    return {'results': [x.stem for x in sorted(base_folder.glob('*')) if x.is_dir()]}


@app.get('/results/{project}', response_model=Results)
async def get_results_for_project(project: str):
    project_folder = LOCAL / project
    return get_sub_folders(project_folder)


@app.get('/results/{project}/{segment}', response_model=Results)
async def get_results_for_segment(project: str, segment: str):
    segment_folder = LOCAL / project / segment
    return get_sub_folders(segment_folder)


@app.get('/results/{project}/{segment}/{video}', response_class=RedirectResponse)
async def get_results_for_video(project: str, segment: str, video: str):
    return RedirectResponse(f'{video}/filtered')


@app.get('/results/{project}/{segment}/{video}/{param}', response_model=VideoData, response_model_exclude_unset=True)
async def get_param_from_video(project: str, segment: str, video: str, param: str, is_critical: bool=Query(False)):
    param_folder = LOCAL / project / segment / video / f'{param}.npy'
    if not param_folder.is_file() or param in {'maxes', 'mins'}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Data no found')
    raw = np.load(param_folder)
    parent_folder = param_folder.parent
    with open(parent_folder / 'metadata.json') as fp:
        meta = loads(fp.read())
    len_ = raw.size
    x = np.linspace(0, len_ * meta['hz'], len_)
    df = pd.DataFrame(np.vstack((x, raw)).T, columns=['x', 'y'])
    df['gesture_point'] = 0
    gesture_points = project_label[project].loc[f'{segment}/{video}', :]
    df.loc[gesture_points.loc[gesture_points['start'] < len_, 'start'], 'gesture_point'] = 1
    df.loc[gesture_points.loc[gesture_points['end'] < len_, 'end'], 'gesture_point'] = -1
    if is_critical:
        maxes = np.load(parent_folder / 'maxes.npy')
        mins = np.load(parent_folder / 'mins.npy')
        diff_size = raw.size - maxes.size
        start = diff_size // 2
        end = start - diff_size + raw.size
        df['critical'] = 0
        df.loc[df.index[start:end], 'critical'] -= maxes.astype(int)
        df.loc[df.index[start:end], 'critical'] += mins.astype(int)
    return {'data': df.to_dict('records')}
