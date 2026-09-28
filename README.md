# BorderFlow AI

Wing Consulting の化学品商社向け・自律型貿易事務エージェントです。

`index.html` を開くと、化学品商社のサンプル案件（CHM-001〜010）が読み込まれます。登録、ステップ完了、与信、船積はブラウザに保存されます。案件一覧の「読み込み」から JSON を投入でき、「書き出し」で同じ形式のファイルを取り出せます。「サンプルを再読込」で初期データに戻せます。

Supabase を使う場合は、先に `supabase/borderflow.sql` を SQL Editor で実行します。画面の「Supabase接続」に Project URL と anon key を入れると、同じデータが `borderflow_workspace` に残ります。未接続のときはブラウザ内だけです。StayPath 用のテーブルとは分けてあります。
