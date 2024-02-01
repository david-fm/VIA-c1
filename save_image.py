from pynput import keyboard
import cv2

def on_press(key):
    if key == keyboard.Key.command:
        #save image
        
        # Stop listener
        return False

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False
# def test_s():
#     while True:
#         if kb.is_pressed('s'):
#             print('s')
#             break

if __name__ == '__main__':
    



    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

    # asyncio.run(main())
    # test_s()