# 결과 그래프 그리기
def fn_draw_graph(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    from matplotlib import pyplot as plt

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='testing acc')
    plt.title('Training and testing accuracy')
    plt.ylim([0.0, 1.2])
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='testing loss')
    plt.title('Training and testing loss')
    plt.ylim([0.0, 1.2])
    plt.legend()
    plt.show()