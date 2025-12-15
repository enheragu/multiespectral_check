

### Cambiar el estilo de Qt (PyQt6)

1) Ver estilos disponibles en tu sistema (se imprimen al lanzar la app ahora):
	- Arranca la app y revisa la consola: se mostrará `Qt styles available: [...]`.
	- También puedes ejecutar en un `python`:
	  ```python
	  from PyQt6.QtWidgets import QApplication, QStyleFactory
	  import sys
	  app = QApplication(sys.argv)
	  print(QStyleFactory.keys())
	  ```

2) Forzar un estilo sin tocar el código: exporta una variable antes de lanzar
	```bash
	export QT_STYLE_OVERRIDE=Fusion   # o el que aparezca en QStyleFactory.keys()
	python -m src.main
	```

3) Estilos instalables: PyQt6 usa plugins de Qt6, no Qt5. En Ubuntu/Debian prueba:
	```bash
	sudo apt install qt6-gtk-platformtheme qt6ct
	```
	Luego elige estilos desde `QT_STYLE_OVERRIDE` o con `qt6ct`.

4) Lo que NO funciona aquí: paquetes `qt5-style-plugins`/`qtcurve` son para Qt5
	y no afectan a PyQt6.