import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv
import os
# .env 파일 로드
load_dotenv()


GMAIL_APP_PASS = os.environ.get("GMAIL_APP_PASS")
SENDER = "min981201@gmail.com"
RECEIVERS = ["min981201@gmail.com"  # 받을 이메일
            # 이메일 추가 하쇼
        ]

def send_email(subject: str, body: str):
    body_html = body.replace("\n", "<br>")
    msg = MIMEText(body_html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SENDER
    msg["To"] = ", ".join(RECEIVERS)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER, GMAIL_APP_PASS)  # 환경변수에서 가져올 것
        smtp.send_message(msg)