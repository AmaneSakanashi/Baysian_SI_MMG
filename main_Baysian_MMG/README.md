## Requirement
- Python <= 3.9（list関数の兼ね合い）
- Numpy (できれば < 2.0 )
- Pandas
- Matplotlib
- Scipy
- meson（Numpy >= 2.0の場合）

高速化のライブラリ、Rayは非推奨（パッケージが複雑）。\
高速化・並列化については検討中です、妙案あれば。

## Compilimg `.f90` fileså

本シミュレータはPythonのみで機能するが、FortranによりMMGモデルの高速化が可能である。しかし、F2pyの仕様上 `.f90` ファイルのコンパイルをあらかじめ行っておく必要があります。

```bash
FC=gnu95 f2py -m mmg_esso -c --f90flags='-O3'  shipsim/ship/esso_osaka/f2py_mmg/mmg_esso_osaka_verctor_input.f90 
```

numpy >= 2.0 を使用する場合、末尾にmesonを明示する方が望ましい。
```bash
--backend meson
```

### For Mac & Linux users
main.pyと同じディレクトリに.soファイルが生成されれば成功。
pythonを実行すれば計算が始まります。

### For Windows users
モジュール名と同様のフォルダが生成されるので（今回 mmg_esso）、
フォルダ内のlibファイル（.dll）をmain.pyのディレクトリに移動。
空のフォルダは削除する。\
この状態でないと、**モジュールが認識されない**ので注意（原因は調査中）。


## Post-process
### main側
log -> cmaフォルダ内のファイルをコピー。\
この作業は計算中でも可能。
### post-process側
opt_resultフォルダ内にペースト。\置き換えることで最新の計算結果を読み込めます。

`num_sample`の値を変えることで、シミュレーションのサンプリング数を変えられます。

