from app import App

def main():
    app = App()

    while True:
        if not app.tick():
            break

if __name__ == '__main__':
    main()
