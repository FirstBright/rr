from chat import Chat
import os

def main():
    if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = "AIzaSyA1g9sPAwHcScfVT_uu_UZWO14JKWWlmdE"
    chat = Chat()   
    response = chat.ask(input("물어보세요: "))
    print(response)


if __name__ == "__main__":
    main()