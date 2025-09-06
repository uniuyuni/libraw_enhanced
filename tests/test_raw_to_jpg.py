#!/usr/bin/env python3
"""
実際のRAWファイルをJPG出力するテストスクリプト
Real RAW file to JPG output test script
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# libraw_enhancedをインポート
try:
    import libraw_enhanced as lre
    print("✅ LibRaw Enhanced imported successfully")
except ImportError as e:
    print(f"❌ Failed to import libraw_enhanced: {e}")
    sys.exit(1)

def find_raw_files():
    """テスト用RAWファイルを検索"""
    test_dir = Path(__file__).parent
    fixtures_dir = test_dir / "fixtures"
    
    raw_extensions = ['.CR2', '.RAF', '.ARW', 
                      '.DNG', '.ORF', '.NEF',
                      '.PEF', '.3FR', '.IIQ']
    
    raw_files = []
    
    # fixtures/を検索
    for search_dir in [fixtures_dir]:
        if search_dir.exists():
            for ext in raw_extensions:
                # 大文字・小文字を区別せずにファイルを検索
                raw_files.extend(search_dir.glob(f"*{ext.lower()}"))
                raw_files.extend(search_dir.glob(f"*{ext.upper()}"))
    
    return sorted(raw_files)

def process_raw_file(raw_path, output_dir):
    """RAWファイルを処理してJPG出力"""
    print(f"\n📸 Processing: {raw_path.name}")
    print("=" * 60)
    
    try:        
        # 複数の処理パラメータでテスト
        test_configs = [
            {
                "name": "AMaZE CPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 32,
                    "demosaic_algorithm": 10,  # AMaZE
                    "use_gpu_acceleration": True
                }
            },
        ]
        """
            {
                "name": "Linear CPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 0,  # Linear
                    "use_gpu_acceleration": False
                }
            },
            {
                "name": "Linear GPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 0,
                    "use_gpu_acceleration": True
                }
            },
            {
                "name": "AMaZE CPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 10,  # AMaZE
                    "use_gpu_acceleration": False
                }
            },
            {
                "name": "AAHD GPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 12,  # AHD
                    "use_gpu_acceleration": True
                }
            },           
            {
                "name": "AHD CPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 3,
                    "use_gpu_acceleration": False
                }
            },
            {
                "name": "AHD GPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 3,
                    "use_gpu_acceleration": True
                }
            },
            {
                "name": "Linear CPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 0,  # Linear
                    "use_gpu_acceleration": False
                }
            },
            {
                "name": "DCB GPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 4,  # DCB
                    "use_gpu_acceleration": True
                }
            },
            {
                "name": "DCB CPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 4,  # DCB
                    "use_gpu_acceleration": False
                }
            },
            {
                "name": "AAHD CPU",
                "params": {
                    "use_camera_wb": True,
                    "half_size": False,
                    "output_bps": 16,
                    "demosaic_algorithm": 12,  # AHD
                    "use_gpu_acceleration": False
                }
            },           
        """        
        results = []
        
        # 各設定で個別にファイルを読み込み直してテスト
        for config in test_configs:
            try:
                print(f"\n🔧 Testing {config['name']} configuration...")
                print(f"📂 Reloading RAW file for {config['name']}...")
                
                # 設定ごとに新しくファイルを読み込み
                with lre.imread(str(raw_path)) as raw_fresh:

                    # 画像情報表示
                    print(f"📏 Dimensions: {raw_fresh.sizes.width}x{raw_fresh.sizes.height}")
                    
                    # カメラ情報取得
                    try:
                        camera_info = raw_fresh.camera_info
                        make = camera_info.get('make', 'Unknown')
                        model = camera_info.get('model', 'Unknown')
                        print(f"📱 Camera: {make} {model}")
                    except Exception as e:
                        print(f"📱 Camera: Info not available ({e})")
                    
                    # 色数情報
                    try:
                        print(f"🎨 Colors: {raw_fresh.num_colors}")
                    except Exception as e:
                        print(f"🎨 Colors: Not available ({e})")

                    # 処理実行
                    process_start = time.time()
                    rgb = raw_fresh.postprocess(**config['params'])
                    process_time = time.time() - process_start
                    
                    print(f"⚡ Processing time: {process_time:.3f}s")
                    print(f"📊 Output shape: {rgb.shape}")
                    print(f"📊 Output dtype: {rgb.dtype}")
                    print(f"📈 Value range: [{np.min(rgb)}, {np.max(rgb)}]")
                    
                    # 8bitに変換
                    if rgb.dtype == np.uint16:
                        rgb_8bit = (rgb / 255).astype(np.uint8)
                    elif rgb.dtype == np.float32:
                        rgb_8bit = (rgb * 255).astype(np.uint8)
                    else:
                        rgb_8bit = rgb.astype(np.uint8)
                    
                    # PIL Imageに変換
                    if len(rgb_8bit.shape) == 3:
                        img = Image.fromarray(rgb_8bit, 'RGB')
                    else:
                        print(f"⚠️ Unexpected image shape: {rgb_8bit.shape}")
                        continue
                    
                    # JPGファイル名生成
                    base_name = raw_path.stem
                    jpg_filename = f"{base_name}_{config['name']}.jpg"
                    jpg_path = output_dir / jpg_filename
                    
                    # JPG保存
                    img.save(jpg_path, 'JPEG', quality=95)
                    print(f"💾 Saved: {jpg_path}")
                    
                    results.append({
                        'config': config['name'],
                        'success': True,
                        'process_time': process_time,
                        'output_path': jpg_path,
                        'shape': rgb.shape,
                        'value_range': [np.min(rgb), np.max(rgb)]
                    })
                
            except Exception as e:
                print(f"❌ Failed {config['name']}: {e}")
                results.append({
                    'config': config['name'],
                    'success': False,
                    'error': str(e)
                })
        
        return results
            
    except Exception as e:
        print(f"❌ Failed to process {raw_path.name}: {e}")
        return []

def main():
    """メイン処理"""
    print("🚀 LibRaw Enhanced - Real RAW File Processing Test")
    print("=" * 70)
    
    # プラットフォーム情報表示
    try:
        platform_info = lre.get_platform_info()
        print(f"🔍 Platform: {platform_info}")
        print(f"🍎 Apple Silicon: {lre.is_apple_silicon()}")
    except Exception as e:
        print(f"⚠️ Could not get platform info: {e}")
    
    # RAWファイル検索
    raw_files = find_raw_files()
    if not raw_files:
        print("❌ No RAW files found in tests/fixtures/")
        return 1
    
    print(f"\n📁 Found {len(raw_files)} RAW files:")
    for raw_file in raw_files:
        print(f"   📸 {raw_file}")
    
    # 出力ディレクトリ作成
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # 各RAWファイルを処理
    results_by_file = {}
    
    for raw_file in raw_files[:3]:  # 最初の3ファイルのみテスト
        results = process_raw_file(raw_file, output_dir)
        if results:
            results_by_file[raw_file.name] = results
    
    # 結果サマリー
    print(f"\n📊 PROCESSING SUMMARY")
    print("=" * 70)
    
    total_success = 0
    total_attempts = 0
    
    for raw_file, results in results_by_file.items():
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        total_success += success_count
        total_attempts += total_count
        
        print(f"📸 {raw_file}: {success_count}/{total_count} configurations successful")
        
        for result in results:
            if result['success']:
                print(f"   ✅ {result['config']}: {result['process_time']:.3f}s")
            else:
                print(f"   ❌ {result['config']}: {result['error']}")
    
    if total_attempts > 0:
        print(f"\n🏆 Overall: {total_success}/{total_attempts} successful ({total_success/total_attempts*100:.1f}%)")
    else:
        print(f"\n🏆 Overall: No processing attempts made")
    
    return 0 if total_success > 0 else 1

if __name__ == "__main__":
    sys.exit(main())