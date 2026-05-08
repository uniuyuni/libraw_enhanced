# Defringe Algorithm Design

## 問題定義

**Lateral Chromatic Aberration (横方向色収差)**とは、レンズの波長別屈折率の違いにより、Red/Blue チャンネルが Green チャンネルと比較して微妙に異なる倍率で撮像される現象。これにより、高コントラスト境界（白い空 vs 暗い葉など）に沿って数ピクセル幅の色付きフリンジが発生する。

### 対象とするフリンジエリア
| 座標 | 内容 | フリンジの特徴 |
|------|------|---------------|
| (6116, 2844) | 葉のフリンジ | 緑/マゼンタ、輝度エッジ沿い |
| (6235, 3018) | カーテン端 | 紫/マゼンタ、白背景エッジ |
| (5939, 2935) | 銀色オブジェクト | 青/紫フリンジ |

### 保護すべきエリア（誤検知防止）
| 座標 | 内容 | 理由 |
|------|------|------|
| (4653, 78) | オレンジハイライト | 正当な色彩、フリンジではない |
| (4080, 3820) | 白い壁 | 均一な低彩度エリア |
| (4610, 4308) | 赤い布 | 正当な彩度の高いオブジェクト |

---

## アルゴリズム調査結果

### darktable Defringe モジュール（参考実装）

darktable の Defringe モジュール (`src/iop/defringe.c`) は以下の原理を使用：

1. Lab 色空間に変換（L は輝度、a/b は色相・彩度）
2. a/b チャンネルを Gaussian blur（フリンジのない「期待される」色を近似）
3. エッジ検出（L チャンネルのグラジエント）
4. 各エッジピクセルで：`|a_orig - a_blur| + |b_orig - b_blur| > threshold` ならフリンジ判定
5. フリンジ判定ピクセルの a/b を blurred 値に置換（L は保持）

**なぜこれが機能するか：**
- CA フリンジは幅 1〜3px の細い異常な色の縞 → Gaussian blur で平均化されて消える
- 正当な色を持つオブジェクト（赤い布など）は面積が十分広いため blur 後も色が残る
- `a_orig ≈ a_blur` → excess ≈ 0 → 補正なし

**なぜ以前のチャンネル excess アプローチが失敗したか：**
- RGB でのダーク側サンプリングは方向依存で不安定
- エッジ方向の推定誤り、lum スケーリングの不正確さ
- 多数の特例ガードが必要で、それ自体が新たなアーティファクトを生む

---

## 設計仕様：Edge-Gated Chroma Bleed Suppression

### アルゴリズム概要

```
入力: RGB float [0,1] (H × W × 3)
出力: RGB float [0,1] (H × W × 3)

1. RGB → Lab 変換（linearize → XYZ → Lab）
2. L チャンネルに Sobel グラジエント → エッジマップ
3. a/b チャンネルに Gaussian blur（半径 radius）
4. 各ピクセル (y, x):
   if edge[y,x] < edge_threshold: 出力 = 入力, continue
   excess = |a[y,x] - blur_a[y,x]| + |b[y,x] - blur_b[y,x]|
   if excess > chroma_threshold:
     a_out = blur_a[y,x]   // フリンジ除去
     b_out = blur_b[y,x]
     L_out = L[y,x]        // 輝度保持
   else:
     a_out = a[y,x]        // 変更なし
     b_out = b[y,x]
     L_out = L[y,x]
5. Lab → RGB 変換
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `radius` | float | 4.0 | Gaussian blur 半径（ピクセル）。値が大きいほど広いフリンジを検出 |
| `edge_threshold` | float | 0.1 | Sobel グラジエント閾値（0〜1）。エッジとみなす最小輝度差 |
| `chroma_threshold` | float | 20.0 | Lab 単位でのクロマ超過閾値。darktable のデフォルトと同じ |
| `mode` | enum | LOCAL | LOCAL: ローカル平均比較、STATIC: 固定閾値 |

darktable の `threshold=20.0` は Lab 空間での a/b 値（-128〜128 の範囲）の差分に相当。

### なぜこの設計か

**Gaussian Blur = フリンジのない「期待される」色の近似**

フリンジのないエリアでは：
- `a[y,x] ≈ mean(a in neighborhood)` → `excess ≈ 0` → 補正なし

フリンジのあるエリアでは：
- フリンジは幅 1〜3px の細い異常色 → 近傍の平均は「背景の白さ（a≈0, b≈0）」
- `a[fringe] = +30`、`blur_a[fringe] ≈ +3` → `excess = 27 > threshold` → 補正

**正当な色（オレンジ、赤い布）の保護**

オレンジハイライト (4653, 78) エリアの場合：
- 近傍も同じオレンジ → `blur_a ≈ a_orig`
- excess ≈ 0 → 補正なし ✅

赤い布 (4610, 4308) エリアの場合：
- 均一な赤色エリア → blur 後も赤い → excess ≈ 0 → 補正なし ✅

---

## 実装計画

### ファイル構成

```
core/
├── cpu_accelerator.cpp/.h    ← defringe() メソッドを追加
├── accelerator.cpp/.h        ← dispatch 関数を追加
├── libraw_wrapper.cpp/.h     ← defringe_numpy() NumPy ラッパーを追加
└── python_bindings.cpp       ← pybind11 バインディング追加

libraw_enhanced/
├── high_level_api.py         ← RawImage.defringe() メソッド + postprocess() 統合
└── constants.py              ← DeFringeMode enum 追加
```

### C++ 実装詳細

#### 1. Lab 変換

```cpp
// sRGB → linear RGB（gamma expand）
float linearize(float x) {
    return x <= 0.04045f ? x / 12.92f
                         : powf((x + 0.055f) / 1.055f, 2.4f);
}

// linear RGB → XYZ (D65 white point)
// [X]   [0.4124564  0.3575761  0.1804375] [R]
// [Y] = [0.2126729  0.7151522  0.0721750] [G]
// [Z]   [0.0193339  0.1191920  0.9503041] [B]

// XYZ → Lab
float f_lab(float t) {
    const float d = 6.0f / 29.0f;
    return t > d*d*d ? cbrtf(t)
                     : t / (3.0f * d * d) + 4.0f / 29.0f;
}
// L = 116 * f(Y/Yn) - 16
// a = 500 * (f(X/Xn) - f(Y/Yn))
// b = 200 * (f(Y/Yn) - f(Z/Zn))
// D65: Xn=0.95047, Yn=1.0, Zn=1.08883
```

#### 2. 分離型 Gaussian Blur

```cpp
// O(W) per pixel using separable 1D convolutions
// horizontal pass → vertical pass
// kernel: Gaussian(sigma) truncated at 3*sigma
```

#### 3. Sobel エッジ検出

```cpp
// L チャンネルに Sobel X/Y フィルタ適用
// edge_map[y][x] = sqrt(Gx^2 + Gy^2) / max_gradient
// normalized to [0, 1]
```

#### 4. フリンジ補正ループ

```cpp
for each pixel (y, x):
    if edge_map[y][x] < edge_threshold: continue
    float excess = fabsf(a[y][x] - blur_a[y][x]) 
                 + fabsf(b[y][x] - blur_b[y][x]);
    if excess > chroma_threshold:
        a_out[y][x] = blur_a[y][x];
        b_out[y][x] = blur_b[y][x];
        // L_out[y][x] は変更なし
```

### Python API

```python
class RawImage:
    def defringe(
        self,
        image: np.ndarray,           # RGB float32 [0,1]
        radius: float = 4.0,
        edge_threshold: float = 0.1,
        chroma_threshold: float = 20.0,
    ) -> np.ndarray:
        """Suppress chromatic aberration fringe using edge-gated chroma bleed suppression.
        
        Based on darktable's defringe algorithm. Detects anomalous chroma at edges
        (compared to Gaussian-blurred reference) and suppresses it.
        
        Args:
            image: RGB float32 array [0,1], shape (H, W, 3)
            radius: Gaussian blur radius in pixels (default 4.0)
            edge_threshold: Minimum edge strength 0-1 (default 0.1)
            chroma_threshold: Lab chroma excess threshold (default 20.0)
        
        Returns:
            Defringe RGB float32 array [0,1], shape (H, W, 3)
        """

    def postprocess(
        self,
        ...,
        defringe: bool = False,
        defringe_radius: float = 4.0,
        defringe_edge_threshold: float = 0.1,
        defringe_chroma_threshold: float = 20.0,
    ) -> np.ndarray:
        """Post-process RAW image. If defringe=True, apply defringe after processing."""
```

---

## テスト計画

### 検証すべきエリア

1. **フリンジ除去確認**（補正されるべき）
   - (6116, 2844): 葉のフリンジ
   - (6235, 3018): カーテン端フリンジ
   - (5939, 2935): 銀オブジェクトフリンジ

2. **誤検知なし確認**（変わらないべき）
   - (4653, 78): オレンジ ± 3px
   - (4080, 3820): 白い壁 ± 3px
   - (4610, 4308): 赤い布 ± 3px

3. **輝度保持確認**（補正前後で L チャンネルが変わらない）

### デフォルトパラメータ根拠

- `radius=4.0`: darktable デフォルト。幅 1〜8px のフリンジを捕捉
- `edge_threshold=0.1`: 全体の約 10〜15% のピクセルがエッジ（適度なフィルタ）
- `chroma_threshold=20.0`: darktable デフォルト。Lab 空間で 20 ≈ 知覚的に明確な色差

---

## 既存アプローチとの比較

| アプローチ | 強み | 弱み |
|-----------|------|------|
| Channel excess (以前の試み) | 方向認識、直感的 | 不安定なダーク側サンプリング、多数の特例ガードが必要 |
| Hue-based suppression | 特定色を正確にターゲット | 色の指定が必要、汎用性低 |
| **Gaussian chroma suppression (本提案)** | シンプル、Lab 空間で数学的に正確、実績あり（darktable） | パラメータチューニングが必要 |
| Geometric channel realignment | 根本原因を修正 | デモザイク後では精度が低下 |

---

## 実装フェーズ

1. **Phase 1**: C++ コア実装（Lab 変換 + Gaussian blur + chroma suppression）
2. **Phase 2**: Python バインディングとテスト
3. **Phase 3**: パラメータチューニング（目標エリアで検証）
4. **Phase 4**: postprocess() への統合
