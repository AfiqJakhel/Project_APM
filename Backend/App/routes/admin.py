import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.background import BackgroundTasks

from app.core.retrain import jalankan_retrain, RETRAIN_STATE

router = APIRouter(tags=["Admin"])

# Skema otentikasi Bearer Token
security = HTTPBearer()

def get_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency untuk mengecek apakah Bearer token yang dikirim
    sama dengan ADMIN_SECRET_TOKEN di environment variables.
    """
    # Secara default jika tidak diset, gunakan string kosong atau lempar error
    admin_secret = os.environ.get("ADMIN_SECRET_TOKEN")
    
    if not admin_secret:
        # Jika server belum mengatur rahasia, tolak semua akses demi keamanan
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_SECRET_TOKEN belum diatur di server environment."
        )
        
    if credentials.credentials != admin_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token admin tidak valid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@router.post("/retrain", summary="Trigger manual untuk retrain model XGBoost")
async def trigger_retrain_manual(
    background_tasks: BackgroundTasks,
    token: str = Depends(get_admin_token)
):
    """
    **Endpoint Admin Terlindungi**
    
    Menjalankan proses `train.py` di background (tidak memblokir request API).
    Proses akan melatih ulang model XGBoost dengan data terbaru di dataset,
    lalu memuat ulang model yang baru ke dalam cache tanpa perlu merestart server.
    """
    if RETRAIN_STATE.get("is_running"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Proses retrain sudah sedang berjalan."
        )
    
    # Kita menggunakan background_tasks agar endpoint segera merespon "OK",
    # sementara proses training berjalan di latar belakang (subprocess).
    background_tasks.add_task(jalankan_retrain)
    
    return {
        "status": "success",
        "message": "Permintaan retrain telah diterima dan sedang diproses di latar belakang.",
        "cek_status_endpoint": "/api/v1/admin/retrain/status"
    }

@router.get("/retrain/status", summary="Cek status proses retrain model")
async def cek_status_retrain(token: str = Depends(get_admin_token)):
    """
    **Endpoint Admin Terlindungi**
    
    Melihat status terakhir dari proses retrain:
    Apakah sedang berjalan, berhasil, kapan terakhir dijalankan, dan jika ada error.
    """
    return {
        "status": "success",
        "data": RETRAIN_STATE
    }
