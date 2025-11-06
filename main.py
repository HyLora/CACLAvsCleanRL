from threading import Thread
import gui
import CACLA

def main():
    cacla_thread : Thread= Thread(target=CACLA.main)
    cacla_thread.daemon = True
    cacla_thread.start()
    gui.main()


if __name__ =="__main__":
    main()