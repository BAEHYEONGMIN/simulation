import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os

load_dotenv()

GMAIL_APP_PASS = os.environ.get("GMAIL_APP_PASS")
SENDER = "min981201@gmail.com"
RECEIVERS = ["min981201@naver.com","min981201@gmail.com","klin0202@naver.com"]
RECEIVERS = ["min981201@naver.com"]

def send_email(subject: str, body: str):
    # 이메일 클라이언트(Gmail 등)에서 줄바꿈이 유지되도록 처리
    body_html = body.replace("\n", "<br>")
    
    msg = MIMEText(body_html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SENDER
    msg["To"] = ", ".join(RECEIVERS)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        print(msg)
        smtp.login(SENDER, GMAIL_APP_PASS)
        smtp.send_message(msg)
