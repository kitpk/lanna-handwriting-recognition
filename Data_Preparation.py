import os
import glob
import xml.etree.ElementTree as ET


def roboflowXML_to_fileTXT(folder_path):
    xmlFire_paths = glob.glob(folder_path + "/*.xml")
    xmlFire_paths = [path.replace("\\", "/") for path in xmlFire_paths]

    folder_out = folder_path + "/TXT"
    if not (os.path.isdir(folder_out)):
        os.mkdir(folder_out)

    for i in range(len(xmlFire_paths)):
        tree = ET.parse(xmlFire_paths[i])
        root = tree.getroot()

        yolos = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name == "helmet":
                name = 1
            xmin = int(obj.find("bndbox").find("xmin").text)
            ymin = int(obj.find("bndbox").find("ymin").text)
            xmax = int(obj.find("bndbox").find("xmax").text)
            ymax = int(obj.find("bndbox").find("ymax").text)
            yolo = f"{name} {xmin} {xmax} {ymin} {ymax}"
            yolos.append(yolo)

        outpath = (
            folder_out
            + xmlFire_paths[i].split(folder_path)[1].split(".xml")[0]
            + ".txt"
        )

        with open(outpath, mode="wt") as f:
            f.write("\n".join(yolos))


if __name__ == "__main__":
    folder_path = "roboflowXML_data/windows_test"
    print("START Data_Preparation")
    roboflowXML_to_fileTXT(folder_path)
    print("END Data_Preparation")
