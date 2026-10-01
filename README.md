# 温度表示・数値認識Webアプリ

画像をアップロードすると、画面中央の `00.0` 形式の数値を認識します。
提供された見本画像に特化した試作版です。

## Streamlit Community Cloudで公開

1. GitHubで新しいリポジトリを作成します。
2. このフォルダ内のファイルをすべてリポジトリへアップロードします。
3. Streamlit Community CloudへGitHubでサインインします。
4. `Create app` を押し、作成したリポジトリを選択します。
5. Main file pathに `app.py` を指定してDeployします。
6. 発行されたURLを共有します。

## ローカル起動

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## 注意

- 現在は、学習画像と同じ画面配置・同程度の解像度を想定しています。
- クラウド公開時、利用者がアップロードした画像はクラウド環境で処理されます。機密画像や個人情報を含む画像を扱う場合は、公開範囲と運用方針を確認してください。
