import { Component, OnInit } from '@angular/core';
import { SaludoService } from './saludo.service';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Router, RouterModule } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule,
    RouterModule],
  template: `<h1>{{ mensaje }}</h1>`,
})
export class AppComponent implements OnInit {
  mensaje = 'El backned esta apagado...';

  constructor(
    private saludoService: SaludoService,
    private router: Router) { }

  ngOnInit() {
    this.saludoService.obtenerSaludo().subscribe((data) => {
      console.log('✅ Backend respondió:', data);
      this.mensaje = data.mensaje;

      if (data?.mensaje) {
        console.log('🚀 Redireccionando al dashboard...');
        this.router.navigate(['/dashboard']);
      }
    });
  }
}
