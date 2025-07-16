import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictService } from '../predict.service';
import { AuthService } from '../auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css'],
  providers: [PredictService], 
})
export class DashboardComponent {
  fileName = '';
  imageSrc: string | ArrayBuffer | null = null;
  imageFile: File | null = null;
  resultado: any = null;
  cargando = false;

  constructor(
    private predictService: PredictService,
    private authService: AuthService,
    private router: Router
  ) {}

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) {
      this.resetForm();
      return;
    }

    this.imageFile = input.files[0];
    this.fileName = this.imageFile.name;

    const reader = new FileReader();
    reader.onload = () => {
      this.imageSrc = reader.result;
      this.resultado = null;
    };
    reader.readAsDataURL(this.imageFile);
  }

  onPredict() {
  if (!this.imageFile) return;

  this.cargando = true;
  this.resultado = null;

  this.predictService.predictImage(this.imageFile).subscribe({
    next: (res) => {
      this.resultado = res;
      this.cargando = false;
    },
    error: (err) => {
      this.resultado = { error: 'Error en la predicción. Intenta de nuevo.' };
      console.error(err);
      this.cargando = false;
    },
  });
}

  onReset() {
    this.resetForm();
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  }

  private resetForm() {
    this.fileName = '';
    this.imageSrc = null;
    this.imageFile = null;
    this.resultado = null;
    this.cargando = false;
  }

  logout() {
    this.authService.logout();
  }
}
