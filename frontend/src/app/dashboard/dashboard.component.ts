import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictService } from '../predict.service'; // Ajusta ruta según tu estructura

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css'],
  providers: [PredictService], // o si usas providedIn:'root' no es necesario
})
export class DashboardComponent {
  fileName = '';
  imageSrc: string | ArrayBuffer | null = null;
  imageFile: File | null = null;  // Guardamos el archivo real
  resultado: any = null;  // Ahora puede ser un objeto JSON
  cargando = false;

  constructor(private predictService: PredictService) { }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) {
      this.fileName = '';
      this.imageSrc = null;
      this.imageFile = null;
      this.resultado = '';
      return;
    }

    this.imageFile = input.files[0];
    this.fileName = this.imageFile.name;

    const reader = new FileReader();
    reader.onload = () => {
      this.imageSrc = reader.result;
      this.resultado = '';
    };
    reader.readAsDataURL(this.imageFile);
  }

  onPredict() {
    if (!this.imageFile) return;

    this.cargando = true;
    this.resultado = '';

    this.predictService.predictImage(this.imageFile).subscribe({
      next: (res) => {
        this.resultado = res;
        this.cargando = false;
      },
      error: (err) => {
        this.resultado = { error: 'Error en la predicción. Intenta de nuevo.' };
        console.error(err);
        this.cargando = false;
      }
    });
  }
  // Método para resetear el formulario y los datos
 onReset() {
    this.fileName = '';
    this.imageSrc = null;
    this.imageFile = null;
    this.resultado = {}; 
    this.cargando = false;
    
    
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    if (fileInput) {
        fileInput.value = '';
    }
}
}


