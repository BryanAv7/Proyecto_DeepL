import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { SaludoService } from '../saludo.service';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root', //app-validacion
  standalone: true,
  imports: [CommonModule, RouterModule, HttpClientModule],
  template: `<h1>{{ mensaje }}</h1>
  <router-outlet></router-outlet>
  `,
})
export class ValidacionComponent implements OnInit {
  mensaje = 'Cargando...';  // Mensaje inicial visible

  constructor(
    private saludoService: SaludoService,
    private router: Router
  ) {}

  ngOnInit() {
    console.log('ValidacionComponent iniciado');

    this.saludoService.obtenerSaludo().subscribe({
      next: (data) => {
        console.log('Respuesta del backend:', data);
        this.mensaje = data.mensaje || 'No se recibió mensaje';

        if (data?.mensaje) {
          this.router.navigate(['/dashboard']);
        }
      },
      error: (err) => {
        console.warn('Error al conectar con backend:', err);
        this.mensaje = 'El backend está apagado...';
      }
    });
  }
}
