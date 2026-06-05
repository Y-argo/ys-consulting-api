# api/core/secret_manager.py
# GCP Secret Managerからcredentialを取得するヘルパー
# ID/PASSはFirestore・ログ・UIに絶対出さない

import json
import os
from typing import Union


def get_secret_json(secret_name: str) -> Union[dict, None]:
    """
    Secret ManagerからJSON形式のsecretを取得する。
    失敗時は例外を投げず、{"blocked": True, "error": "..."} を返す。
    ID/PASSは絶対にログに出さない。
    """
    if not secret_name:
        return {"blocked": True, "error": "credential_secret_nameが指定されていません"}

    try:
        from google.cloud import secretmanager
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-consulting-ai")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("utf-8")
    except Exception as e:
        # secretが存在しない・権限なし・ネットワークエラー等
        # 例外メッセージにID/PASSが含まれないよう、secret_nameのみ記録
        print(f"[secret_manager] secret取得失敗: secret_name={secret_name} error_type={type(e).__name__}", flush=True)
        return {"blocked": True, "error": f"Secret Manager取得失敗: {type(e).__name__}"}

    try:
        creds = json.loads(payload)
    except json.JSONDecodeError:
        return {"blocked": True, "error": "SecretのJSON parseに失敗しました。形式を確認してください"}

    if not isinstance(creds, dict):
        return {"blocked": True, "error": "Secretはdictである必要があります"}

    # 必須キー確認（値はログに出さない）
    required_keys = ["username", "password"]
    missing = [k for k in required_keys if k not in creds]
    if missing:
        return {"blocked": True, "error": f"Secretに必須キーがありません: {', '.join(missing)}"}

    return creds


def save_secret_json(secret_name: str, data: dict) -> dict:
    """
    Secret ManagerへJSON形式のsecretを保存する。
    既存secretがある場合はadd_secret_versionで上書き。
    ID/PASSは絶対にログに出さない。
    """
    import json
    from google.cloud import secretmanager
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-consulting-ai")
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project_id}"
    secret_path = f"{parent}/secrets/{secret_name}"

    payload_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")

    # secretが存在するか確認
    try:
        client.get_secret(request={"name": secret_path})
        secret_exists = True
    except Exception:
        secret_exists = False

    try:
        if not secret_exists:
            # 新規作成
            client.create_secret(request={
                "parent": parent,
                "secret_id": secret_name,
                "secret": {"replication": {"automatic": {}}},
            })
            print(f"[secret_manager] secret作成: secret_name={secret_name}", flush=True)

        # バージョン追加（上書き）
        client.add_secret_version(request={
            "parent": secret_path,
            "payload": {"data": payload_bytes},
        })
        print(f"[secret_manager] secret保存完了: secret_name={secret_name}", flush=True)
        return {"status": "saved"}

    except Exception as e:
        print(f"[secret_manager] secret保存失敗: secret_name={secret_name} error_type={type(e).__name__}", flush=True)
        raise RuntimeError(f"Secret Manager保存失敗: {type(e).__name__}")
