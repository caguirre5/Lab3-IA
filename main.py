import flet as ft
from laboratorio3 import lab3
from libreria import modelo


def main(page: ft.Page):
    def button_clicked(e):
        result = ''
        accuracy = ''
        t.value = f"Your favorite color is:  {cg.value}"
        if cg.value == 'own':
            result, accuracy = lab3(tb.value)
        elif cg.value == 'sklearn':
            result, accuracy = modelo(tb.value)
        else:
            result = 'No seleccionaste ningun modelo'

        t.value = 'El mensaje que has ingresado ha sido clasificado como {}'.format(
            result)
        t2.value = "Precisión del modelo: {:.2f}%".format(accuracy * 100)
        page.update()

    tb = ft.TextField(
        label="Escriba un mensaje que desee clasificar",
    )
    t = ft.Text(color='red', size=20)
    t2 = ft.Text(color='green', size=20)
    b = ft.ElevatedButton(text='Submit', on_click=button_clicked)
    cg = ft.RadioGroup(content=ft.Column([
        ft.Radio(value="own", label="Libreria propia"),
        ft.Radio(value="sklearn", label="Libreria sklearn")]))

    page.add(ft.Text("Seleccione el modelo que quiere usar"), tb, cg, b, t, t2)


ft.app(target=main)
