import torch
import copy
import torchvision.transforms as transforms
import time
import numpy as np
import ntpath
import os

def test(model, test_loader, device, class_number, need_output = False, per_image = False, need_output_time = False):
    model.eval()
    confusion_matrix = np.zeros((class_number, class_number))
    confusion_matrix = confusion_matrix.astype(int)
    start_time = None
    with torch.no_grad():
        correct = 0
        total = 0
        batch_num = 0
        batch_size = test_loader.batch_size
        for images, labels in test_loader:
            if not per_image:
                images = images.to(device)
                labels = labels.to(device)

                if need_output_time:
                    start_time = time.time()

                outputs = model(images)

                if need_output_time:
                    total_time = (time.time() - start_time)
                    print('predict time: {}ms'.format(total_time * 1000))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(len(images)):
                    actual_class = labels[i]
                    predicted_class = predicted[i]
                    confusion_matrix[predicted_class, actual_class] += 1
                    if need_output:
                        if predicted_class != actual_class:
                            print('{};predicted={},correct={}'.format(test_loader.dataset.imgs[batch_num * batch_size + i][0],
                                                                            test_loader.dataset.classes[predicted_class],
                                                                            test_loader.dataset.classes[actual_class]))
            else:
                for i in range(len(images)):
                    image = images[i]
                    label = labels[i]

                    image = image.unsqueeze(0)
                    label = label.unsqueeze(0)

                    image = image.to(device)
                    label = label.to(device)

                    output = model(image)

                    _, predicted = torch.max(output.data, 1)
                    total += 1
                    correct += (predicted == label).sum().item()

                    if need_output:
                        if predicted[0] != label[0]:
                            print('{}, predicted = {}, correct = {}'.format(test_loader.dataset.imgs[batch_num * batch_size + i][0],
                                                                            test_loader.dataset.classes[predicted[0]],
                                                                            test_loader.dataset.classes[label[0]]))
            batch_num+=1

    acc_net_current = 100 * correct / total
    return (acc_net_current, confusion_matrix)


def train(model, criterion, optimizer, train_loader,test_loader, num_epochs, device, class_number, need_output = True):
    max_accuracy = 0;
    max_accuracy_model = copy.deepcopy(model)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if need_output:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        acc, _ = test(model, test_loader, device, class_number)
        if need_output:
            print('Epoch {}, Acc: {}'.format(epoch + 1, acc))
        if acc > max_accuracy:
            max_accuracy = acc
            max_accuracy_model = copy.deepcopy(model)

        if epoch % 50 == 0:
            torch.save(max_accuracy_model.state_dict(), 'max_accuracy_model.ckpt')
    return max_accuracy_model, max_accuracy



def testTopResults(model, test_loader, device, class_number, topCount, need_output = False):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        batch_num = 0
        batch_size = test_loader.batch_size
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pytorch_output = outputs.data.cpu().numpy()
            labels_out = labels.data.cpu().numpy()

            total += labels.size(0)
            for i in range(len(images)):
                actual_class = labels_out[i]

                arr = pytorch_output[i]
                idx = arr.argsort()[-topCount:][::-1]
                predicted_class = predicted[i]

                isCorrect = False
                for j in range(topCount):
                    if idx[j] == actual_class:
                        correct = correct+1
                        isCorrect = True
                        break

                if need_output:
                    if not isCorrect:
                        print('{};predicted={},correct={}'.format(
                            test_loader.dataset.imgs[batch_num * batch_size + i][0],
                            test_loader.dataset.classes[predicted_class],
                            test_loader.dataset.classes[actual_class]))

            batch_num+=1

    acc_net_current = 100 * correct / total
    return acc_net_current


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def getResult(model, test_loader, device):
    model.eval()
    f = open("result.txt", "w+")
    f.write("id_code,diagnosis\n")
    with torch.no_grad():
        correct = 0
        total = 0
        batch_num = 0
        batch_size = test_loader.batch_size
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(images)):
                predicted_class = predicted[i]
                no_ext_path = os.path.splitext(test_loader.dataset.imgs[batch_num * batch_size + i][0])[0]
                name = path_leaf(no_ext_path)

                f.write('{},{}\n'.format(name, predicted_class))

            batch_num += 1

    f.close()
